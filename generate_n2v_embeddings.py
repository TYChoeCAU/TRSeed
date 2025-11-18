# generate_n2v_embeddings.py  (multi p,q sweep)

import os
import argparse
import time
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec

def parse_float_list(s):
    return [float(x) for x in str(s).split(',') if str(x).strip()]

def load_graph_from_preprocessed_files(data_path):
    """
    ./data/<dataset>/{train.txt}만 사용해 그래프를 생성합니다.
    각 라인은: user item1 item2 ...  (공백 또는 탭 구분 둘 다 허용)
    """
    print(f"[load] ONLY train.txt from {data_path} (to avoid leakage)")
    fp = os.path.join(data_path, 'train.txt')
    if not os.path.isfile(fp):
        raise FileNotFoundError(f"not found: {fp}")

    edges = []
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()  # 공백/탭 자동 분리
            if len(parts) < 2: 
                continue
            u = int(parts[0])
            for it in parts[1:]:
                v = int(it)
                edges.append((f"u{u}", f"i{v}"))

    G = nx.Graph(edges)
    print(f"[load] nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")
    return G

def run_node2vec(G, dim, walk_len, num_walks, p, q, seed=42, workers=1):
    n2v = Node2Vec(
        G,
        dimensions=dim,
        walk_length=walk_len,
        num_walks=num_walks,
        workers=workers,      # Windows 안전을 위해 1 권장
        p=p,
        q=q,
        seed=seed,
        quiet=False
    )
    # 기본 파라미터는 Word2Vec 기본값과 유사
    model = n2v.fit(window=10, min_count=1, batch_words=4)
    return model

def fmt(x):
    """파일명용 간단 포맷(불필요한 0 제거)"""
    s = f"{x:.6g}"
    return s

def main():
    ap = argparse.ArgumentParser(description="Generate multiple Node2Vec embeddings by sweeping p and q.")
    ap.add_argument('--dataset', required=True,
                    choices=['ali','aminer','ml-1m','gowalla','amazon','yelp2018'])
    ap.add_argument('--data_root', default='./data', help="root dir of datasets")
    ap.add_argument('--out_dir',   default='./n2v', help="where to save .kv files")
    ap.add_argument('--dimensions', type=int, default=512)
    ap.add_argument('--walk_length', type=int, default=80)
    ap.add_argument('--num_walks',   type=int, default=20)
    ap.add_argument('--p_list', type=str, default='0.25,0.5,1,5,10,100')
    ap.add_argument('--q_list', type=str, default='0.25,0.5,1,5,10,100')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=1, help=">=2는 Linux만 권장")
    ap.add_argument('--force', action='store_true', help="overwrite existing files")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 그래프 로드(한 번만)
    data_path = os.path.join(args.data_root, args.dataset)
    G = load_graph_from_preprocessed_files(data_path)

    # 2) 스윕 목록
    P = parse_float_list(args.p_list)
    Q = parse_float_list(args.q_list)
    print(f"[sweep] p_list={P}, q_list={Q}")
    print(f"[cfg] dim={args.dimensions}, walk_length={args.walk_length}, num_walks={args.num_walks}")

    # 3) 모든 조합 실행
    for p in P:
        for q in Q:
            tag = f"{args.dataset}_p{fmt(p)}_q{fmt(q)}_d{args.dimensions}.kv"
            out_path = os.path.join(args.out_dir, tag)

            if (not args.force) and os.path.exists(out_path):
                print(f"[skip] exists -> {out_path}")
                continue

            print(f"[run] p={p}, q={q} -> {out_path}")
            t0 = time.time()
            model = run_node2vec(
                G,
                dim=args.dimensions,
                walk_len=args.walk_length,
                num_walks=args.num_walks,
                p=p, q=q,
                seed=args.seed,
                workers=args.workers
            )
            model.wv.save_word2vec_format(out_path)
            print(f"[done] saved ({time.time()-t0:.1f}s) -> {out_path}")

    print("[all done]")

if __name__ == "__main__":
    main()
