import os
import random
import torch
import numpy as np
from time import time
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from utils.helper import early_stopping

import optuna
import joblib
import datetime

# ---- global device (default) ----
device = torch.device("cpu")


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1, K=1, n_items=0):
    def sampling(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for _ in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(
        sampling(entity_pairs, train_pos_set, n_negs * K)
    ).to(device)
    return feed_dict


def opt_objective(trial, args, train_cf, user_dict, n_params, norm_mat, deg, outdeg, search_space):
    """One Optuna grid trial (GridSampler + categorical)."""
    args.dim = trial.suggest_categorical('dim', search_space['dim'])
    args.l2 = trial.suggest_categorical('l2', search_space['l2'])
    args.context_hops = trial.suggest_categorical('context_hops', search_space['context_hops'])

    valid_res_list = []
    for seed in range(args.runs):
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        valid_best_result = main(args, seed, train_cf, user_dict, n_params, norm_mat, deg, outdeg)
        valid_res_list.append(valid_best_result)
    return float(np.mean(valid_res_list))


def main(args, run, train_cf, user_dict, n_params, norm_mat, deg, outdeg):
    """define model"""
    from TRSeed_APPNP import APPNP
    model = APPNP(n_params, args, norm_mat, deg).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.lr}])

    n_items = n_params['n_items']
    cur_best_pre_0 = 0.0
    stopping_step = 0
    best_epoch = 0
    print("start training ...")

    hyper = {"dim": args.dim, "l2": args.l2, "hops": args.context_hops, "lamb": getattr(args, "lamb", None)}
    print("Start hyper parameters: ", hyper)

    for epoch in range(args.epoch):
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_ = train_cf[index].to(device)

        """training"""
        model.train()
        loss, s = 0.0, 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf_):
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  args.n_negs,
                                  args.K,
                                  n_items)
            batch_loss, _, _ = model(batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += float(batch_loss.detach().cpu().item())
            s += args.batch_size
        train_e_t = time()
        print('loss:', round(loss, 2), "time: ", round(train_e_t - train_s_t, 2), 's')

        if epoch % args.step == 0:
            """testing"""
            # PrettyTable 헤더 원상복구 (precision, hit_ratio 포함)
            train_res = PrettyTable()
            train_res.field_names = ["Phase", "Epoch", "training time(s)", "testing time(s)",
                                     "Loss", "recall", "ndcg", "precision", "hit_ratio"]

            model.eval()
            test_s_t = time()
            test_ret, user_result, deg_recall, deg_recall_mean = test(model, user_dict, n_params, deg, mode='test')
            os.makedirs('./logs', exist_ok=True)
            with open('./logs/' + args.gnn + '_deg_recall_mean_' + str(args.context_hops) + '_' + str(args.dataset) + '.txt', 'w') as f:
                for deg_ in deg_recall_mean:
                    f.write(str(deg_) + '\t' + str(deg_recall_mean[deg_]) + '\n')
            test_e_t = time()

            train_res.add_row(
                ["Test", epoch,
                 round(train_e_t - train_s_t, 2), round(test_e_t - test_s_t, 2), round(loss, 2),
                 test_ret['recall'], test_ret['ndcg'], test_ret.get('precision', []), test_ret.get('hit_ratio', [])]
            )

            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time()
                valid_ret, user_result, deg_recall, deg_recall_mean = test(model, user_dict, n_params, deg, mode='valid')
                test_e_t = time()
                train_res.add_row(
                    ["Valid", epoch,
                     round(train_e_t - train_s_t, 2), round(test_e_t - test_s_t, 2), round(loss, 2),
                     valid_ret['recall'], valid_ret['ndcg'], valid_ret.get('precision', []), valid_ret.get('hit_ratio', [])]
                )

            print(train_res)

            # @K 인덱싱 원복: recall[0] 유지 (첫 번째 K에 대한 값 사용)
            cur_best_pre_0, stopping_step, should_stop = early_stopping(
                valid_ret['recall'][0], cur_best_pre_0, stopping_step, expected_order='acc', flag_step=10
            )
            if valid_ret['recall'][0] == cur_best_pre_0:
                best_epoch = epoch
            if should_stop:
                break

            """save weight"""
            if valid_ret['recall'][0] == cur_best_pre_0 and args.save:
                os.makedirs(args.out_dir, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(args.out_dir, f'{args.dataset}_{args.dim}_{args.context_hops}_{args.l2}_' + args.gnn + '.ckpt')
                )
                best_epoch = epoch
        else:
            print('using time %.4fs, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss))

    print('early stopping at %d, recall@20:%.4f, best_epoch at %d' % (epoch, cur_best_pre_0, best_epoch))
    print("Seed:", run)
    print("End hyper parameters: ", hyper)
    print(f"Best valid_ret['recall']: ", cur_best_pre_0)
    return cur_best_pre_0


if __name__ == '__main__':
    # 1) read args
    args = parse_args()

    # 2) time/device
    s = datetime.datetime.now()
    print("time of start: ", s)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # 3) build dataset
    train_cf, user_dict, n_params, norm_mat, deg, outdeg = load_data(args)

    # ---- 안전한 (N,2) long 텐서 변환 ----
    arr = np.asarray(train_cf)
    if arr.ndim != 2 or arr.shape[1] < 2:
        # list of pairs 등 방어적 폴백
        arr = np.array([[cf[0], cf[1]] for cf in train_cf], dtype=np.int64)
    else:
        arr = arr[:, :2].astype(np.int64, copy=False)
    train_cf = torch.from_numpy(arr)

    # 4) 실행 모드: disable_grid가 켜져 있으면 고정 하이퍼 1회 실행
    use_fixed = getattr(args, 'disable_grid', False)
    if use_fixed:
        steps = args.context_hops
        search_space = {
            'dim': [args.dim],
            'context_hops': [int(steps)],
            'l2': [args.l2],
        }
        n_trials = 1
        print(f"[run] fixed hypers -> steps={steps}, dim={args.dim}, l2={args.l2}")
    else:
        # 데이터셋별 grid
        all_search_spaces = {
            'aminer':   {'dim': [512], 'context_hops': [2], 'l2': [1e-3]},
            'ml-1m':    {'dim': [512], 'context_hops': [1], 'l2': [1e-2]},
            'ml-100k':  {'dim': [512], 'context_hops': [1], 'l2': [1e-2]},
            'ali':      {'dim': [512], 'context_hops': [3], 'l2': [1e-3]},
            'amazon':   {'dim': [512], 'context_hops': [1], 'l2': [1e-2]},
            'gowalla':  {'dim': [512], 'context_hops': [4], 'l2': [1e-3]},
            'yelp2018': {'dim': [512], 'context_hops': [3], 'l2': [1e-3]},
        }
        if args.dataset not in all_search_spaces:
            raise ValueError(f"Unknown dataset: {args.dataset}. "
                             f"Choose one of {list(all_search_spaces.keys())}")
        search_space = all_search_spaces[args.dataset]
        n_trials = int(np.prod([len(v) for v in search_space.values()])) or 1

    print("search_space: ", search_space)
    print("n_trials (grid size): ", n_trials)

    # 5) Optuna GridSampler (maximize)
    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler(search_space),
        direction="maximize"
    )
    study.optimize(lambda trial: opt_objective(
        trial, args, train_cf, user_dict, n_params, norm_mat, deg, outdeg, search_space
    ), n_trials=n_trials)

    # 6) save study
    joblib.dump(study, f'{args.dataset}_{args.dim}_{args.context_hops}_{args.l2}_study_' + args.gnn + '.pkl')

    e = datetime.datetime.now()
    print(study.best_trial.params)
    print("time of end: ", e)
