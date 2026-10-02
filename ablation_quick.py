# -*- coding: utf-8 -*-
"""MAPTrans 快速消融验证脚本：手动划分（不写 fold 文件），支持传任意开关组合。

用法示例（MAPTrans conda 环境）：
  python ablation_quick.py --epochs 120 --num_metapaths 4 --use_view_trans 1 --use_inter_cat 1 \
      --use_raw_feat 1 --use_prot_mlp 1 --use_multi_sim_feat 1
"""
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as fn
import dgl
from data_preprocess import get_data, data_processing, dgl_similarity_graph, dgl_heterograph
from MAPTrans import MAPTrans
from metric import get_metric
from ap_loss import soft_ap_loss

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='F-dataset')
p.add_argument('--epochs', type=int, default=120)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--weight_decay', type=float, default=1e-3)
p.add_argument('--random_seed', type=int, default=1234)
p.add_argument('--num_metapaths', type=int, default=4)
p.add_argument('--use_view_trans', type=int, default=1)
p.add_argument('--use_view_pool', type=int, default=1)
p.add_argument('--use_orig_view', type=int, default=0)
p.add_argument('--use_inter_cat', type=int, default=1)
p.add_argument('--use_raw_feat', type=int, default=1)
p.add_argument('--use_prot_mlp', type=int, default=1)
p.add_argument('--use_multi_sim_feat', type=int, default=1)
p.add_argument('--use_gdc', type=int, default=1)
p.add_argument('--use_adv', type=int, default=1)
p.add_argument('--use_fuse_ssl', type=int, default=1)
p.add_argument('--use_lowrank_ap', type=int, default=1)
p.add_argument('--use_grad_balance', type=int, default=1)
p.add_argument('--use_ap', type=int, default=1)
p.add_argument('--use_ssl', type=int, default=1)
p.add_argument('--use_gating', type=int, default=1)
p.add_argument('--use_ce_weight', type=int, default=1)
flags = p.parse_args()

args = argparse.Namespace(
    dataset=flags.dataset, k_fold=2, epochs=flags.epochs, lr=flags.lr, weight_decay=flags.weight_decay,
    random_seed=flags.random_seed, neighbor=20, negative_rate=1.0, dropout=0.4,
    gt_layer=2, gt_head=2, gt_out_dim=200, hgt_layer=2, hgt_head=8, hgt_in_dim=64,
    hgt_head_dim=25, hgt_out_dim=200, tr_layer=2, tr_head=4,
    use_gdc=flags.use_gdc, gdc_alpha=0.05, gdc_order=3,
    use_adv=flags.use_adv, adv_eps=0.01, adv_warmup=50,
    use_fuse_ssl=flags.use_fuse_ssl,
    use_lowrank_ap=flags.use_lowrank_ap, ap_rank=2,
    use_grad_balance=flags.use_grad_balance, gb_warmup=50,
    num_metapaths=flags.num_metapaths, use_view_trans=flags.use_view_trans,
    use_view_pool=flags.use_view_pool, use_orig_view=flags.use_orig_view,
    use_inter_cat=flags.use_inter_cat, use_raw_feat=flags.use_raw_feat,
    use_prot_mlp=flags.use_prot_mlp, use_multi_sim_feat=flags.use_multi_sim_feat,
    use_ap=flags.use_ap, use_ssl=flags.use_ssl,
    use_gating=flags.use_gating, use_ce_weight=flags.use_ce_weight,
)
args.data_dir = 'data/' + args.dataset + '/'

random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

data, adj_matrices = get_data(args)
for key in adj_matrices:
    adj_matrices[key] = torch.tensor(adj_matrices[key], dtype=torch.float32)
args.drug_number = data['drug_number']
args.disease_number = data['disease_number']
args.protein_number = data['protein_number']
args.drug_feat_dim = data['drugfeature'].shape[1]
args.disease_feat_dim = data['diseasefeature'].shape[1]
args.protein_feat_dim = data['proteinfeature'].shape[1]
if args.use_multi_sim_feat:
    args.drug_sim_dim = data['drf'].shape[1] + data['drg'].shape[1]
    args.disease_sim_dim = data['dip'].shape[1] + data['dig'].shape[1]

data = data_processing(data, args)
X = data['all_drdi']
Y = data['all_label']
idx = np.arange(len(X))
rng = np.random.RandomState(args.random_seed)
rng.shuffle(idx)
n_test = len(idx) // 5
test_idx, train_idx = idx[:n_test], idx[n_test:]
X_train = torch.LongTensor(X[train_idx])
Y_train = torch.LongTensor(np.expand_dims(Y[train_idx], 1))
X_test = torch.LongTensor(X[test_idx])
Y_test = Y[test_idx].flatten()

drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)
drdipr_graph, data, meta_paths_drug, meta_paths_disease = dgl_heterograph(data, X_train.numpy(), args)
homo_graph_cpu = dgl.to_homogeneous(drdipr_graph)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
drdr_graph = drdr_graph.to(device)
didi_graph = didi_graph.to(device)
drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)
X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)

model = MAPTrans(args).to(device)
opt = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

pos_count = Y_train.sum().item()
neg_count = len(Y_train) - pos_count
if args.use_ce_weight:
    ce_weight = torch.tensor([1.0, (neg_count / pos_count) ** 0.5], device=device)
else:
    ce_weight = torch.ones(2, device=device)
cross_entropy = torch.nn.CrossEntropyLoss(weight=ce_weight)

best_auc, best_aupr = 0.0, 0.0
for ep in range(args.epochs):
    model.train()
    opt.zero_grad()
    ssl_labels = Y_train if args.use_ssl else None
    dr, score, ssl = model(drdr_graph, didi_graph, homo_graph_cpu, drug_feature, disease_feature, protein_feature,
                           X_train, meta_paths_drug, meta_paths_disease, adj_matrices, labels=ssl_labels)
    ce = cross_entropy(score, Y_train.flatten())
    ssl_weight = model.ssl_lambda * min(1.0, ep / 50)
    prob = fn.softmax(score, dim=-1)[:, 1]
    if args.use_ap:
        ap_loss = soft_ap_loss(prob, Y_train.flatten().float(), tau=0.03, neg_samples=20000,
                               hard_ratio=1.0, sample=X_train, rank=args.ap_rank if args.use_lowrank_ap else 0)
        ap_weight = 0.05 + (0.50 - 0.05) * min(1.0, max(0.0, (ep - 30) / 270))
    else:
        ap_loss = torch.tensor(0.0, device=device)
        ap_weight = 0.0
    loss = ce + ssl_weight * ssl + ap_weight * ap_loss
    # 梯度清零必须放在 backward 之前：--use_ckpt 开启（本脚本默认）时，
    # MAPTrans._score_mlp 的 _CheckpointedMLPBlock 在 backward 中会把段内 MLP
    # 参数的梯度直接累加到 p.grad，backward 之后再 zero_grad 会把本次有效梯度
    # 一并清掉，导致 optimizer.step() 空转（模型不更新）。与 train_DDA.py 保持一致。
    opt.zero_grad()
    loss.backward()
    opt.step()

    if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
        with torch.no_grad():
            model.eval()
            dr, tscore, _ = model(drdr_graph, didi_graph, homo_graph_cpu, drug_feature, disease_feature,
                                  protein_feature, X_test, meta_paths_drug, meta_paths_disease, adj_matrices)
        tprob = fn.softmax(tscore, dim=-1)[:, 1].cpu().numpy()
        tscore_arg = torch.argmax(tscore, dim=-1).cpu().numpy()
        AUC, AUPR, _, _, _, _, _ = get_metric(Y_test, tscore_arg, tprob)
        if AUPR > best_aupr:
            best_aupr, best_auc = AUPR, AUC
        if (ep + 1) % 20 == 0 or ep == args.epochs - 1:
            print('ep %d: auc=%.4f aupr=%.4f | best auc=%.4f aupr=%.4f'
                  % (ep + 1, AUC, AUPR, best_auc, best_aupr), flush=True)

print('FINAL best_auc=%.4f best_aupr=%.4f' % (best_auc, best_aupr))
