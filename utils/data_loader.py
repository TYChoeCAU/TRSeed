# utils/data_loader.py
import os
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ---- globals (main / model에서 사용) ----
n_users = 0
n_items = 0
dataset = ''
train_user_set = defaultdict(list)
train_item_set = defaultdict(list)
test_user_set  = defaultdict(list)
valid_user_set = defaultdict(list)


# ---------- 파일 포맷 자동 감지 ----------
def _detect_format(path, max_probe_lines=200):
    """
    빠른 라인 프리뷰로 파일 포맷을 추정.
    반환:
      mode: 'pairs' | 'pairs_extra' | 'adj'
      sep : 'whitespace' (split()) -- 실제 파싱 시 모두 공백 분리로 처리
    규칙:
      - 모든 비어있지 않은 줄의 토큰 수가 동일하면:
          == 2      -> 'pairs'
          >= 3      -> 'pairs_extra' (rating/timestamp 등 고정 열이 더 있음)
      - 줄마다 토큰 수가 다르면 -> 'adj' (user + 가변 개수의 item들)
    """
    counts = []
    with open(path, 'r', encoding='utf-8') as f:
        for _ in range(max_probe_lines):
            line = f.readline()
            if not line:
                break
            toks = line.strip().split()  # 공백/탭 모두 처리
            if not toks:
                continue
            counts.append(len(toks))
    if not counts:
        return 'pairs', 'whitespace'  # 빈 파일이면 무난한 기본

    uniq = set(counts)
    if len(uniq) == 1:
        c = counts[0]
        return ('pairs' if c == 2 else 'pairs_extra'), 'whitespace'
    else:
        return 'adj', 'whitespace'


# ---------- 로더: 어떤 형식이든 (N,2) (u,i)로 변환 ----------
def read_cf_auto(file_name, dedup=True):
    """
    파일 형식을 자동 감지하여 (u,i) 쌍 ndarray(int32)로 반환.
    - whitespace split으로 공백/탭 모두 처리.
    - 'adj'  : user item1 item2 ... -> 모든 (u, item_k) 생성
    - 'pairs': user item
    - 'pairs_extra': user item rating timestamp ... -> user, item만 사용
    """
    mode, _ = _detect_format(file_name)
    pairs = []

    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            toks = line.strip().split()
            if not toks:
                continue

            if mode == 'adj':
                u = int(toks[0])
                # 남은 모든 열을 item으로 간주
                # set으로 중복 제거 후 (u,i) 생성
                for it in set(toks[1:]):
                    pairs.append((u, int(it)))
            else:
                # pairs / pairs_extra : 앞의 두 열만 사용
                if len(toks) < 2:
                    continue
                u, i = int(toks[0]), int(toks[1])
                pairs.append((u, i))

    if dedup:
        # 매우 큰 파일에서도 파이썬 set이 훨씬 안정적
        pairs = list(set(pairs))

    if not pairs:
        return np.empty((0, 2), dtype=np.int32)

    arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    # 이후 파이프라인과 호환 위해 int32로 반환 (torch에서 long으로 올림)
    return arr.astype(np.int32, copy=False)


# ---------- 통계/사전 구성 ----------
def statistics(train_data, valid_data, test_data):
    """
    (N,2)의 (u,i) 배열을 받아 n_users/n_items를 계산하고,
    user별/ item별 인접 리스트 딕셔너리를 채운다.
    또한, '사용자/아이템 ID 공간이 합쳐져 있는 케이스'를 자동 보정한다.
    """
    global n_users, n_items, train_user_set, train_item_set, test_user_set, valid_user_set

    # 기존 누적 지우기 (재호출 안전)
    train_user_set.clear(); train_item_set.clear()
    test_user_set.clear();  valid_user_set.clear()

    # 기본 추정
    if train_data.size:
        max_u = int(train_data[:, 0].max())
        min_u = int(train_data[:, 0].min())
    else:
        max_u, min_u = -1, 0

    if valid_data.size:
        max_u = max(max_u, int(valid_data[:, 0].max()))
        min_u = min(min_u, int(valid_data[:, 0].min()))
    if test_data.size:
        max_u = max(max_u, int(test_data[:, 0].max()))
        min_u = min(min_u, int(test_data[:, 0].min()))

    n_users_guess = max_u + 1 if max_u >= 0 else 0

    # item id의 최소/최대를 계산
    def _min_max_item(arr):
        if arr.size == 0:
            return None, None
        return int(arr[:, 1].min()), int(arr[:, 1].max())

    mins, maxs = [], []
    for a in (train_data, valid_data, test_data):
        mn, mx = _min_max_item(a)
        if mn is not None:
            mins.append(mn); maxs.append(mx)

    if mins:
        min_item_id = min(mins)
        max_item_id = max(maxs)
    else:
        min_item_id, max_item_id = 0, -1

    # ---- merged-space 탐지 및 보정 ----
    # (일부 데이터는 item id가 n_users부터 시작하는 "합쳐진" 인덱싱)
    # 이 경우 item을 모두 (id - n_users_guess)로 쉬프트한다.
    merged_space = (min_item_id >= n_users_guess and n_users_guess > 0)

    if merged_space:
        # 모든 split에 대해 아이템 id 쉬프트
        if train_data.size: train_data[:, 1] -= n_users_guess
        if valid_data.size: valid_data[:, 1] -= n_users_guess
        if test_data.size:  test_data[:, 1] -= n_users_guess

        n_users = n_users_guess
        # 쉬프트 후 다시 최대 아이템 id 계산
        def _max_item(a):
            return int(a[:, 1].max()) if a.size else -1
        n_items = max(_max_item(train_data), _max_item(valid_data), _max_item(test_data)) + 1
    else:
        # 일반적인 0-base item id
        n_users = n_users_guess
        n_items = max_item_id + 1 if max_item_id >= 0 else 0

    # ---- 딕셔너리 구성 ----
    for u_id, i_id in train_data:
        u, i = int(u_id), int(i_id)
        train_user_set[u].append(i)
        train_item_set[i].append(u)
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in valid_data:
        valid_user_set[int(u_id)].append(int(i_id))

    print('n_users: ', n_users, '\tn_items: ', n_items)
    print('n_train: ', len(train_data), '\tn_test: ', len(test_data), '\tn_valid: ', len(valid_data))
    print('n_inters: ', len(train_data) + len(test_data) + len(valid_data))


# ---------- 정규화 인접행렬 ----------
def build_sparse_graph(data_cf):
    """
    Bi‑normalized Laplacian (D^{-1/2} A D^{-1/2})를 반환.
    입력 (u,i) 쌍에서 양방향 블록 [[0,R],[R^T,0]]를 구성한다.
    """
    def _bi_norm_lap(adj):
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    cf = np.asarray(data_cf, dtype=np.int64)
    # [0, n_items) -> [n_users, n_users+n_items)
    ui = cf.copy()
    ui[:, 1] = ui[:, 1] + n_users

    iu = ui[:, [1, 0]]  # flip for R^T
    edges = np.concatenate([ui, iu], axis=0)  # [[0,R],[R^T,0]]

    vals = np.ones(len(edges), dtype=np.float32)
    mat = sp.coo_matrix((vals, (edges[:, 0], edges[:, 1])), shape=(n_users + n_items, n_users + n_items))
    # deg (in/out)도 반환 (모델 코드에서 centrality 초기화에 사용)
    indeg = np.array(mat.sum(1))  # (N,1)
    outdeg = np.array(mat.sum(0)) # (1,N)
    return _bi_norm_lap(mat), indeg, outdeg


# ---------- 엔트리: main에서 호출 ----------
def load_data(model_args):
    """
    args.data_path/{dataset}/ 하위의 train.txt / valid.txt / test.txt를 읽는다.
    - 각 파일은 공백/탭/가변열 형식 모두 허용(자동 감지).
    - 반환 형식/키는 기존 main/모델과 동일.
    """
    global dataset
    args = model_args
    dataset = args.dataset
    directory = os.path.join(args.data_path, dataset) + os.sep

    # 파일 경로
    train_path = os.path.join(directory, 'train.txt')
    valid_path = os.path.join(directory, 'valid.txt')
    test_path  = os.path.join(directory, 'test.txt')

    print('reading train and test user-item set ...')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test file not found: {test_path}")

    # valid가 없으면 빈 set으로 처리 (원본 코드와 달리 안전)
    has_valid = os.path.exists(valid_path)

    train_cf = read_cf_auto(train_path, dedup=True)
    test_cf  = read_cf_auto(test_path,  dedup=True)
    valid_cf = read_cf_auto(valid_path,  dedup=True) if has_valid else np.empty((0,2), dtype=np.int32)

    # 안전한 (N,2) 보장 이후 통계/사전 구성
    statistics(train_cf, valid_cf, test_cf)

    print('building the adj mat ...')
    norm_mat, indeg, outdeg = build_sparse_graph(train_cf)

    n_params = {'n_users': int(n_users), 'n_items': int(n_items)}
    user_dict = {
        'train_item_set': train_item_set,
        'train_user_set': train_user_set,
        'valid_user_set': valid_user_set if args.dataset not in ['yelp2018', 'gowalla'] else None,
        'test_user_set':  test_user_set,
    }
    print('loading over ...')
    return train_cf, user_dict, n_params, norm_mat, indeg, outdeg
