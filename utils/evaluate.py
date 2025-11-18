# utils/evaluate.py
import math
import numpy as np
import torch
from contextlib import nullcontext

@torch.no_grad()
def _ndcg_at_k(topk_items, gt_set, k):
    dcg = 0.0
    for i, iid in enumerate(topk_items[:k]):
        if iid in gt_set:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt_set), k)))
    return (dcg / idcg) if idcg > 0 else 0.0

def _get_eval_users(user_dict, n_users, mode):
    # 평가할 유저 집합(테스트 아이템 있는 유저만)
    tgt = user_dict['test_user_set'] if mode == 'test' else user_dict.get('valid_user_set') or user_dict['test_user_set']
    eval_users = [u for u in range(n_users) if len(tgt.get(u, [])) > 0]
    return eval_users

def _vectorized_mask(scores, batch_users, user_dict, device):
    """배치 단위로 노출 아이템 -inf 마스킹 (한 번에 인덱싱)"""
    rows, cols = [], []
    for bi, u in enumerate(batch_users):
        excl = set(user_dict['train_user_set'].get(u, []))
        vu = user_dict.get('valid_user_set', None)
        if vu is not None:
            excl |= set(vu.get(u, []))
        if excl:
            rows.append(torch.full((len(excl),), bi, device=device, dtype=torch.long))
            cols.append(torch.tensor(list(excl), device=device, dtype=torch.long))
    if rows:
        rows = torch.cat(rows, dim=0)
        cols = torch.cat(cols, dim=0)
        scores[rows, cols] = float('-inf')

def _update_topk(top_scores, top_idx, block_scores, block_start, K):
    """전역 top‑K와 블록 top‑K를 합쳐 다시 top‑K로 갱신"""
    bs = block_scores.shape[0]
    bk = min(K, block_scores.shape[1])
    # 블록 내부 top‑K
    b_scores, b_idx = torch.topk(block_scores, k=bk, dim=1)  # [B, bk]
    b_idx = b_idx + block_start                              # 전역 아이템 인덱스로 오프셋
    # 결합
    cand_scores = torch.cat([top_scores, b_scores], dim=1)   # [B, K+bk]
    cand_idx    = torch.cat([top_idx,    b_idx],    dim=1)   # [B, K+bk]
    new_scores, new_pos = torch.topk(cand_scores, k=K, dim=1)
    new_idx = torch.gather(cand_idx, 1, new_pos)
    return new_scores, new_idx

@torch.no_grad()
def test(model, user_dict, n_params, deg, mode='test',
         Ks=[20], score_user_batch=4096, score_item_block=0, use_fp16=True):
    """
    Fast all-ranking evaluation.
    - score_user_batch: 유저 배치 크기
    - score_item_block: 0이면 아이템 전체 한 번에, >0이면 블록 단위(예: 50000)
    - use_fp16       : GPU에서 FP16/TF32로 가속
    """
    device = next(model.parameters()).device
    model.eval()

    # 1) 임베딩 생성 & 디바이스/정밀도 세팅
    with torch.no_grad():
        u_emb, i_emb = model.generate(split=True)
    u_emb = u_emb.to(device, non_blocking=True)
    i_emb = i_emb.to(device, non_blocking=True)

    # Half/TF32
    amp_ctx = nullcontext()
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        if use_fp16:
            u_emb = u_emb.half()
            i_emb = i_emb.half()
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    Ks = list(Ks) if isinstance(Ks, (list, tuple)) else [int(Ks)]
    Kmax = max(Ks)

    eval_users = _get_eval_users(user_dict, n_users, mode)
    recalls = {K: [] for K in Ks}
    ndcgs   = {K: [] for K in Ks}

    # degree별 결과(기본 K=Ks[0])
    user_degrees = np.zeros(n_users, dtype=np.int64)
    for u, items in user_dict['train_user_set'].items():
        user_degrees[int(u)] = len(items)
    deg_recall = {}  # degree -> list of recall@Ks[0]

    with amp_ctx:
        for s in range(0, len(eval_users), score_user_batch):
            batch_users = eval_users[s:s + score_user_batch]
            if not batch_users:
                continue

            U = u_emb[batch_users]  # [B, d]

            if score_item_block and score_item_block < n_items:
                # (A) 아이템 블록 집계: 메모리/시간 절충
                top_scores = torch.full((len(batch_users), Kmax), float('-inf'), device=device, dtype=U.dtype)
                top_idx    = torch.full((len(batch_users), Kmax), -1, device=device, dtype=torch.long)
                for st in range(0, n_items, score_item_block):
                    ed = min(st + score_item_block, n_items)
                    block = i_emb[st:ed]                                  # [bI, d]
                    scores_block = U @ block.t()                           # [B, bI]
                    # 블록 내 노출 마스킹 (로컬 열 인덱스로 변환)
                    rows, cols = [], []
                    for bi, u in enumerate(batch_users):
                        excl = user_dict['train_user_set'].get(u, [])
                        vu = user_dict.get('valid_user_set', None)
                        if vu is not None:
                            excl = set(excl) | set(vu.get(u, []))
                        if excl:
                            # st<=iid<ed 인 것만 로컬 인덱스로
                            loc = [iid - st for iid in excl if st <= iid < ed]
                            if loc:
                                rows.append(torch.full((len(loc),), bi, device=device, dtype=torch.long))
                                cols.append(torch.tensor(loc, device=device, dtype=torch.long))
                    if rows:
                        rows = torch.cat(rows, 0); cols = torch.cat(cols, 0)
                        scores_block[rows, cols] = float('-inf')

                    top_scores, top_idx = _update_topk(top_scores, top_idx, scores_block, st, Kmax)

                # 최종 후보 (전역 top‑K)
                topk_all = top_idx
            else:
                # (B) 아이템 전체 한 번에 (GPU 메모리 여유 있을 때)
                scores = U @ i_emb.t()                                     # [B, I]
                _vectorized_mask(scores, batch_users, user_dict, device)
                topk_all = torch.topk(scores, k=min(Kmax, n_items), dim=1).indices  # [B, Kmax]

            # 메트릭
            for bi, u in enumerate(batch_users):
                gt = set(user_dict['test_user_set'].get(u, []))
                if not gt:
                    continue
                cand = topk_all[bi].tolist()
                for K in Ks:
                    hits = len([iid for iid in cand[:K] if iid in gt])
                    rc = hits / max(1, len(gt))
                    recalls[K].append(rc)
                    ndcgs[K].append(_ndcg_at_k(cand, gt, K))
                # degree별 (K=Ks[0])
                d = int(user_degrees[u])
                deg_recall.setdefault(d, []).append(recalls[Ks[0]][-1])

    # 요약
    res = {
        'recall': [float(np.mean(recalls[K])) if recalls[K] else 0.0 for K in Ks],
        'ndcg'  : [float(np.mean(ndcgs[K]))   if ndcgs[K]   else 0.0 for K in Ks],
        'precision': [0.0 for _ in Ks],  # (필요 시 추가 계산)
        'hit_ratio': [0.0 for _ in Ks],
    }
    # degree 평균
    deg_recall_mean = {k: float(np.mean(v)) if len(v) else 0.0 for k, v in deg_recall.items()}

    # main.py와 호환을 위해 동일한 튜플 반환
    user_result = None
    return res, user_result, deg_recall, deg_recall_mean
