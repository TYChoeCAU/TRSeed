import torch
import torch.nn as nn
from gensim.models import KeyedVectors
import scipy.sparse as sp


class GraphConv(nn.Module):
    def __init__(
        self,
        n_hops,
        n_users,
        n_items,
        interact_mat,
        deg,
        e=1e-7,
        edge_dropout_rate=0.5,
        mess_dropout_rate=0.1,
        t_u=2, t_i=2,
        device=torch.device("cpu"),
        interact_mat_coo=None    
    ):
        super(GraphConv, self).__init__()

        self.n_hops = n_hops
        self.n_users = n_users
        self.n_items = n_items
        self.edge_dropout_rate = float(edge_dropout_rate)
        self.mess_dropout_rate = float(mess_dropout_rate)

        self.interact_mat = interact_mat
        self.interact_mat_coo = interact_mat_coo

        self.dropout = nn.Dropout(p=self.mess_dropout_rate) 
        self.device = device

        degree = torch.sigmoid(torch.log(torch.from_numpy(deg) + e)).to(self.device)
        t_u_prior, t_i_prior = torch.split(degree, [n_users, n_items], dim=0)

        init = nn.init.xavier_uniform_
        self.user_t = nn.Parameter(init(torch.empty(self.n_users, 1, device=self.device)), requires_grad=True)
        self.item_t = nn.Parameter(init(torch.empty(self.n_items, 1, device=self.device)), requires_grad=True)
        with torch.no_grad():
            self.user_t.copy_(t_u_prior)
            self.item_t.copy_(t_i_prior)

    def _sparse_dropout(self, x_coo: torch.Tensor, rate: float = 0.5):

        keep_prob = 1.0 - float(rate)
        if keep_prob <= 0.0:
            return x_coo
        nnz = x_coo._nnz()
        rand = keep_prob + torch.rand(nnz, device=x_coo.device)
        mask = torch.floor(rand).to(torch.bool)

        i = x_coo.indices()[:, mask]
        v = x_coo.values()[mask]
        v = v * (1.0 / keep_prob)

        out = torch.sparse_coo_tensor(i, v, size=x_coo.shape, device=x_coo.device)
        return out.coalesce()

    def forward(self, user_embed, item_embed, mess_dropout=True, edge_dropout=True):

        all_embed = torch.cat([user_embed, item_embed], dim=0)  
        agg_embed = all_embed

        u_t = self.user_t                     
        i_t = self.item_t                    
        one_u = torch.ones_like(u_t)
        one_i = torch.ones_like(i_t)

       
        u_weight0 = u_t
        i_weight0 = i_t
        ego = torch.cat([u_weight0 * user_embed, i_weight0 * item_embed], dim=0)
        embs = [ego]  

        use_dropout = (self.training and edge_dropout and self.edge_dropout_rate > 0.0)
        if use_dropout:
            A = self._sparse_dropout(
                self.interact_mat_coo if self.interact_mat_coo is not None
                else self.interact_mat.to_sparse_coo(),
                self.edge_dropout_rate
            )
            use_coo = True
        else:
            A = self.interact_mat  
            use_coo = False

        decay_u = (one_u - u_t)  
        decay_i = (one_i - i_t)  
        pow_u = one_u.clone()
        pow_i = one_i.clone()

        for _k in range(1, self.n_hops + 1):
            if use_coo:
                side = torch.sparse.mm(A, agg_embed)
            else:
                side = A.matmul(agg_embed)

            u_side, i_side = torch.split(side, [self.n_users, self.n_items], dim=0)

            pow_u = pow_u * decay_u
            pow_i = pow_i * decay_i

            u_k = u_side * u_t * pow_u
            i_k = i_side * i_t * pow_i
            side_k = torch.cat([u_k, i_k], dim=0)  

            agg_embed = side
            if mess_dropout and self.training:
                agg_embed = self.dropout(agg_embed)

            embs.append(side_k)

        embs = torch.stack(embs, dim=1)
        return embs[:self.n_users, :], embs[self.n_users:, :]


class APPNP(nn.Module):

    def __init__(self, data_config, args_config, adj_mat, deg):
        super(APPNP, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        self.deg = deg
        self.args = args_config

        self.e = args_config.e
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs
        self.ns = args_config.ns
        self.K = args_config.K
        self.heatkernel = args_config.heatkernel

        self.device = torch.device("cuda:0") if getattr(args_config, "cuda", False) else torch.device("cpu")
        self.t_u = args_config.t_u
        self.t_i = args_config.t_i

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)  
        self.item_embed = nn.Parameter(self.item_embed)

        self.gcn = self._init_model()

    def _init_weight(self):
        init = nn.init.xavier_uniform_

        self.user_embed = init(torch.empty(self.n_users, self.emb_size, device=self.device))
        self.item_embed = init(torch.empty(self.n_items, self.emb_size, device=self.device))

        self.n2v_emb = None
        if hasattr(self.args, 'n2v_path') and self.args.n2v_path:
            n2v_path = self.args.n2v_path
            print(f"[APPNP] Loading Node2Vec embeddings from: {n2v_path}")
            n2v_model = KeyedVectors.load_word2vec_format(n2v_path)
            n2v_dim = n2v_model.vector_size

            if n2v_dim != self.emb_size:
                raise ValueError(
                    f"[APPNP] SUM mixing requires equal dims: E={self.emb_size}, Z={n2v_dim}. "
                    f"Please retrain Node2Vec to dim={self.emb_size} or switch to a concat path."
                )

            w = torch.zeros((self.n_users + self.n_items, n2v_dim), dtype=torch.float32, device=self.device)
            for i in range(self.n_users):
                key = f"u{i}"
                if key in n2v_model:
                    w[i] = torch.from_numpy(n2v_model[key]).to(self.device)
            for j in range(self.n_items):
                key = f"i{j}"
                if key in n2v_model:
                    w[self.n_users + j] = torch.from_numpy(n2v_model[key]).to(self.device)
            self.n2v_emb = nn.Embedding.from_pretrained(w, freeze=True)
            print("[APPNP] Node2Vec loaded (frozen).")

        self.sparse_norm_adj_csr = self._to_torch_csr(self.adj_mat).to(self.device)
        self.sparse_norm_adj_coo = self.sparse_norm_adj_csr.to_sparse_coo().coalesce()

    def _init_model(self):
        return GraphConv(
            n_hops=self.context_hops,
            n_users=self.n_users,
            n_items=self.n_items,
            interact_mat=self.sparse_norm_adj_csr,
            deg=self.deg,
            e=self.e,
            edge_dropout_rate=self.edge_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate,
            t_u=self.t_u,
            t_i=self.t_i,
            device=self.device,
            interact_mat_coo=self.sparse_norm_adj_coo
        )

    def _to_torch_csr(self, X: sp.spmatrix) -> torch.Tensor:
        X = X.tocsr().astype('float32')
        crow = torch.from_numpy(X.indptr.astype('int64'))
        col = torch.from_numpy(X.indices.astype('int64'))
        val = torch.from_numpy(X.data.astype('float32'))
        return torch.sparse_csr_tensor(crow, col, val, size=X.shape)

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  

        cur_user = self.user_embed
        cur_item = self.item_embed
        if self.n2v_emb is not None:
            lamb = float(self.args.lamb)
            n2v = self.n2v_emb.weight  
            z_user, z_item = torch.split(n2v, [self.n_users, self.n_items], dim=0)
            cur_user = self.user_embed + lamb * z_user
            cur_item = self.item_embed + lamb * z_item

        user_gcn_emb, item_gcn_emb = self.gcn(
            cur_user, cur_item,
            edge_dropout=self.edge_dropout,
            mess_dropout=self.mess_dropout
        )

        if self.ns == 'rns':  
            neg_gcn_embs = item_gcn_emb[neg_item[:, :self.K]]
        else:
            neg_gcn_embs = []
            for k in range(self.K):
                neg_gcn_embs.append(
                    self.negative_sampling(
                        user_gcn_emb, item_gcn_emb,
                        user, neg_item[:, k * self.n_negs: (k + 1) * self.n_negs],
                        pos_item
                    )
                )
            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)

        return self.create_bpr_loss(user_gcn_emb[user], item_gcn_emb[pos_item], neg_gcn_embs)

    def negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)  

        seed = torch.rand(batch_size, 1, p_e.shape[1], 1, device=p_e.device)  
        n_e = item_gcn_emb[neg_candidates]  
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e

        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  
        return neg_items_emb_[[[i] for i in range(batch_size)], range(neg_items_emb_.shape[1]), indices, :]

    def pooling(self, embeddings):
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # 'final'
            return embeddings[:, -1, :]

    def generate(self, split=True):
        cur_user = self.user_embed
        cur_item = self.item_embed
        if self.n2v_emb is not None:
            lamb = float(self.args.lamb)
            n2v = self.n2v_emb.weight
            z_user, z_item = torch.split(n2v, [self.n_users, self.n_items], dim=0)
            cur_user = self.user_embed + lamb * z_user
            cur_item = self.item_embed + lamb * z_item

        user_gcn_emb, item_gcn_emb = self.gcn(cur_user, cur_item, edge_dropout=False, mess_dropout=False)
        user_gcn_emb = self.pooling(user_gcn_emb)
        item_gcn_emb = self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):

        B = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb)                               
        pos_e = self.pooling(pos_gcn_embs)                             
        neg_e = self.pooling(neg_gcn_embs.view(-1,
                                               neg_gcn_embs.shape[2],
                                               neg_gcn_embs.shape[3])).view(B, self.K, -1)  

        pos_scores = torch.sum(u_e * pos_e, dim=1)                     
        neg_scores = torch.sum(u_e.unsqueeze(1) * neg_e, dim=-1)       

        mf_loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(1)).sum(dim=1)))

        regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                      + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                      + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2
        emb_loss = self.decay * regularize / B

        return mf_loss + emb_loss, mf_loss, emb_loss
