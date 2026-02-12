import os
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
dataset = ''
train_user_set = defaultdict(list)
train_item_set = defaultdict(list)
test_user_set  = defaultdict(list)
valid_user_set = defaultdict(list)


def _detect_format(path, max_probe_lines=200):
    counts = []
    with open(path, 'r', encoding='utf-8') as f:
        for _ in range(max_probe_lines):
            line = f.readline()
            if not line:
                break
            toks = line.strip().split()  
            if not toks:
                continue
            counts.append(len(toks))
    if not counts:
        return 'pairs', 'whitespace'  

    uniq = set(counts)
    if len(uniq) == 1:
        c = counts[0]
        return ('pairs' if c == 2 else 'pairs_extra'), 'whitespace'
    else:
        return 'adj', 'whitespace'


def read_cf_auto(file_name, dedup=True):
    mode, _ = _detect_format(file_name)
    pairs = []

    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            toks = line.strip().split()
            if not toks:
                continue

            if mode == 'adj':
                u = int(toks[0])
                for it in set(toks[1:]):
                    pairs.append((u, int(it)))
            else:
                if len(toks) < 2:
                    continue
                u, i = int(toks[0]), int(toks[1])
                pairs.append((u, i))

    if dedup:
        pairs = list(set(pairs))

    if not pairs:
        return np.empty((0, 2), dtype=np.int32)

    arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return arr.astype(np.int32, copy=False)


def statistics(train_data, valid_data, test_data):

    global n_users, n_items, train_user_set, train_item_set, test_user_set, valid_user_set

    train_user_set.clear(); train_item_set.clear()
    test_user_set.clear();  valid_user_set.clear()

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

    merged_space = (min_item_id >= n_users_guess and n_users_guess > 0)

    if merged_space:
        if train_data.size: train_data[:, 1] -= n_users_guess
        if valid_data.size: valid_data[:, 1] -= n_users_guess
        if test_data.size:  test_data[:, 1] -= n_users_guess

        n_users = n_users_guess
        def _max_item(a):
            return int(a[:, 1].max()) if a.size else -1
        n_items = max(_max_item(train_data), _max_item(valid_data), _max_item(test_data)) + 1
    else:
        n_users = n_users_guess
        n_items = max_item_id + 1 if max_item_id >= 0 else 0

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


def build_sparse_graph(data_cf):

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
    indeg = np.array(mat.sum(1))  # (N,1)
    outdeg = np.array(mat.sum(0)) # (1,N)
    return _bi_norm_lap(mat), indeg, outdeg


def load_data(model_args):

    global dataset
    args = model_args
    dataset = args.dataset
    directory = os.path.join(args.data_path, dataset) + os.sep

    train_path = os.path.join(directory, 'train.txt')
    valid_path = os.path.join(directory, 'valid.txt')
    test_path  = os.path.join(directory, 'test.txt')

    print('reading train and test user-item set ...')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test file not found: {test_path}")

    has_valid = os.path.exists(valid_path)

    train_cf = read_cf_auto(train_path, dedup=True)
    test_cf  = read_cf_auto(test_path,  dedup=True)
    valid_cf = read_cf_auto(valid_path,  dedup=True) if has_valid else np.empty((0,2), dtype=np.int32)

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
