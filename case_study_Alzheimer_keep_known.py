# -*- coding: utf-8 -*-
"""
阿尔茨海默病（AD）案例：非冷启动设置（与 MAPTrans 案例口径一致）

实验设置（与 case_study_Alzheimer.py 的“全屏蔽冷启动”相反）：
  1. 训练集【保留全部】已知药物-疾病关联，AD 的已知正样本【不删除、不屏蔽】，
     模型在训练时见过 AD 已有的治疗关联；
  2. AD 的未知 (drug, AD) 配对【照常参与全局负采样】——训练集构造与
     train_DDA.py 全量训练完全一致，不做任何目标疾病列剔除；
  3. 训练完成后，对【每个药物 × AD】候选配对打分，后处理阶段把数据集中
     已知的 AD 正样本（known=yes）全部剔除、不参与排序；
  4. 在仅剩下的 unknown（known=no）候选集合内按预测概率降序取 Top-K（默认 10）。

用途：模拟“疾病已有治疗、希望补充更多新药”的真实重定位场景
      （非冷启动；对应论文案例研究的对照口径，勿与全屏蔽口径的 P 值比较）。

用法（在 zuizhong 目录、MAPTrans 环境内执行）：
  python case_study_Alzheimer_keep_known.py --dataset C-dataset
  python case_study_Alzheimer_keep_known.py --dataset C-dataset --epochs 600 --topk 10

说明：
  - 无独立验证集：AD 已知正样本已在训练中，不能再用作“测试代理”选 epoch，
    因此使用【最终 epoch 权重】打分（train_DDA.py 同款训练全程）。
  - 训练超参数与创新模块开关默认值与 case_study_Alzheimer.py / train_DDA.py 完全一致。
  - 药物名/DrugBank ID 映射复用 case_study_Alzheimer.load_drug_id_and_name
    （DrugInformation.csv 行序 = 药物索引序）。
"""

import os
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as fn
import torch.optim as optim
import dgl

from data_preprocess import (get_data, data_processing, dgl_similarity_graph,
                             dgl_heterograph)
from MAPTrans import MAPTrans
from case_study_Alzheimer import find_ad_index, load_drug_id_and_name


def main():
    parser = argparse.ArgumentParser(
        description='AD 案例（非冷启动）：全量训练(含 AD 已知药) + 仅对 unknown 候选 Top-K')
    # ---- 案例研究专属参数 ----
    parser.add_argument('--dataset', default='F-dataset', help='F-dataset / C-dataset / B-dataset')
    parser.add_argument('--epochs', type=int, default=600, help='训练轮数（与 train_DDA.py 默认 600 一致）')
    parser.add_argument('--topk', type=int, default=10, help='输出 Top-K 候选药物（unknown 内）')
    parser.add_argument('--output_dir', default='Result/CaseStudy', help='结果输出目录')
    # ---- 与 train_DDA.py 一致的训练超参数 ----
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--neighbor', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--gt_layer', type=int, default=2)
    parser.add_argument('--gt_head', type=int, default=2)
    parser.add_argument('--gt_out_dim', type=int, default=200)
    parser.add_argument('--hgt_layer', type=int, default=2)
    parser.add_argument('--hgt_head', type=int, default=8)
    parser.add_argument('--hgt_in_dim', type=int, default=64)
    parser.add_argument('--hgt_head_dim', type=int, default=25)
    parser.add_argument('--hgt_out_dim', type=int, default=200)
    parser.add_argument('--tr_layer', type=int, default=2)
    parser.add_argument('--tr_head', type=int, default=4)
    # ---- 创新模块开关（默认 = train_DDA.py，与完整模型一致）----
    parser.add_argument('--use_gdc', type=int, default=1)
    parser.add_argument('--gdc_alpha', type=float, default=0.05)
    parser.add_argument('--gdc_order', type=int, default=3)
    parser.add_argument('--use_adv', type=int, default=1)
    parser.add_argument('--adv_eps', type=float, default=0.01)
    parser.add_argument('--adv_warmup', type=int, default=50)
    parser.add_argument('--use_fuse_ssl', type=int, default=1)
    parser.add_argument('--use_lowrank_ap', type=int, default=1)
    parser.add_argument('--ap_rank', type=int, default=2)
    parser.add_argument('--use_grad_balance', type=int, default=1)
    parser.add_argument('--gb_warmup', type=int, default=50)
    parser.add_argument('--num_metapaths', type=int, default=4)
    parser.add_argument('--use_view_trans', type=int, default=1)
    parser.add_argument('--use_view_pool', type=int, default=1)
    parser.add_argument('--use_orig_view', type=int, default=0)
    parser.add_argument('--use_inter_cat', type=int, default=1)
    parser.add_argument('--use_raw_feat', type=int, default=1)
    parser.add_argument('--use_prot_mlp', type=int, default=1)
    parser.add_argument('--use_multi_sim_feat', type=int, default=1)
    parser.add_argument('--use_ckpt', type=int, default=1)
    parser.add_argument('--use_ap', type=int, default=1)
    parser.add_argument('--use_ssl', type=int, default=1)
    parser.add_argument('--use_gating', type=int, default=1)
    parser.add_argument('--use_ce_weight', type=int, default=1)
    args = parser.parse_args()

    args.data_dir = 'data/' + args.dataset + '/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 统一随机种子
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    # ========== 数据加载 ==========
    data, adj_matrices = get_data(args)
    for key in adj_matrices:                      # 邻接转 CPU tensor（forward 内按需移 GPU）
        adj_matrices[key] = torch.tensor(adj_matrices[key], dtype=torch.float32)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']
    args.drug_feat_dim = data['drugfeature'].shape[1]
    args.disease_feat_dim = data['diseasefeature'].shape[1]
    args.protein_feat_dim = data['proteinfeature'].shape[1]
    if getattr(args, 'use_multi_sim_feat', 1):
        args.drug_sim_dim = data['drf'].shape[1] + data['drg'].shape[1]
        args.disease_sim_dim = data['dip'].shape[1] + data['dig'].shape[1]

    data = data_processing(data, args)            # 生成 drs/dis 与全量样本(含所有候选负样本)

    ad_idx = find_ad_index(args.dataset, args.data_dir)
    n_drugs, n_dis = data['drug_number'], data['disease_number']
    print('Dataset:', args.dataset, '| drugs=%d diseases=%d proteins=%d | AD disease index=%d'
          % (n_drugs, n_dis, data['protein_number'], ad_idx))

    # 数据集中已知 (drug, AD) 阳性（本设置下【保留在训练集中】，仅在后处理排序时剔除）
    known_ad = set(int(d) for (d, dis) in data['drdi'] if int(dis) == ad_idx)

    # ---- 训练样本：全量保留（非冷启动）----
    # AD 已知正样本照常作为正例；AD 未知配对照常参与全局负采样，不做任何屏蔽。
    all_pairs = np.asarray(data['all_drdi'], dtype=np.int64)
    all_label = np.asarray(data['all_label'], dtype=np.int64)
    X_tr = all_pairs
    Y_tr = all_label
    X_train = torch.LongTensor(X_tr).to(device)
    Y_train = torch.LongTensor(np.expand_dims(Y_tr, 1)).to(device)
    n_ad_rows = int((all_pairs[:, 1] == ad_idx).sum())
    print('[non-cold-start] train pairs = %d (其中 AD 列配对 %d 条保留在训练集)'
          % (len(X_tr), n_ad_rows))

    # 候选集：仅 unknown（known=yes 直接剔除，不参与排序）
    cand_idx = [i for i in range(n_drugs) if i not in known_ad]
    cand = torch.LongTensor([[i, ad_idx] for i in cand_idx]).to(device)
    print('scoring candidates = %d (all unknown; known-AD positives kept in training = %d)'
          % (len(cand_idx), len(known_ad)))

    # ========== 建图（一次，训练期不变）==========
    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)
    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)
    drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)

    drdipr_graph, data, meta_paths_drug, meta_paths_disease = \
        dgl_heterograph(data, X_tr, args)
    homo_graph_cpu = dgl.to_homogeneous(drdipr_graph)

    # ========== 模型 ==========
    model = MAPTrans(args).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    # CE 类别加权（sqrt）与 AP 动态负采样系数（逻辑与 train_DDA.py 相同）
    pos_count = int(Y_train.sum().item())
    neg_count = len(Y_train) - pos_count
    actual_ratio = neg_count / pos_count
    print('Fold-like stats: pos=%d neg=%d ratio=1:%.1f' % (pos_count, neg_count, actual_ratio))
    if getattr(args, 'use_ce_weight', 1):
        ce_weight = torch.tensor([1.0, actual_ratio ** 0.5], device=device)
    else:
        ce_weight = torch.ones(2, device=device)
    cross_entropy = torch.nn.CrossEntropyLoss(weight=ce_weight)

    ap_lambda = 0.50
    ap_warmup_epochs = 30
    ap_tau = 0.03
    ap_hard_ratio = 1.0
    ap_neg_ratio_start = max(2.0, actual_ratio * 0.03)
    ap_neg_ratio_end = actual_ratio * 0.5
    ap_neg_ramp_epochs = 500

    # 候选打分函数（最终权重调用一次）
    @torch.no_grad()
    def score_candidates(model):
        model.eval()
        _, out, _ = model(drdr_graph, didi_graph, homo_graph_cpu, drug_feature,
                          disease_feature, protein_feature, cand, meta_paths_drug,
                          meta_paths_disease, adj_matrices)
        return fn.softmax(out, dim=-1)[:, 1]

    # ========== 训练（与 train_DDA.py 全量训练完全一致，无独立验证集）==========
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        ssl_labels = Y_train if getattr(args, 'use_ssl', 1) else None
        dr_representation, train_score, ssl_loss = model(
            drdr_graph, didi_graph, homo_graph_cpu, drug_feature, disease_feature,
            protein_feature, X_train, meta_paths_drug, meta_paths_disease,
            adj_matrices, labels=ssl_labels)
        ce_loss = cross_entropy(train_score, torch.flatten(Y_train))
        ssl_weight = model.ssl_lambda * min(1.0, epoch / 50)
        train_prob = fn.softmax(train_score, dim=-1)[:, 1]

        neg_ratio = ap_neg_ratio_start + (ap_neg_ratio_end - ap_neg_ratio_start) * \
            min(1.0, epoch / ap_neg_ramp_epochs)
        dyn_neg = min(int(pos_count * neg_ratio), neg_count)
        ap_rank = args.ap_rank if args.use_lowrank_ap else 0
        if getattr(args, 'use_ap', 1):
            from ap_loss import soft_ap_loss
            ap_loss = soft_ap_loss(train_prob, torch.flatten(Y_train).float(),
                                   tau=ap_tau, neg_samples=dyn_neg,
                                   hard_ratio=ap_hard_ratio, sample=X_train, rank=ap_rank)
        else:
            ap_loss = torch.tensor(0.0, device=device)

        # AP 权重两阶段 ramp（0.05 -> 0.50）
        if getattr(args, 'use_ap', 1):
            if epoch < ap_warmup_epochs:
                ap_weight = 0.05 * (epoch / ap_warmup_epochs)
            else:
                ap_weight = 0.05 + (ap_lambda - 0.05) * min(1.0, (epoch - ap_warmup_epochs) / 270)
        else:
            ap_weight = 0.0

        # 方向 D：FGSM 对抗扰动（两次前向，逻辑同 train_DDA.py）
        if args.use_adv and epoch >= args.adv_warmup:
            loss_clean = ce_loss + ssl_weight * ssl_loss + ap_weight * ap_loss
            g_d, g_r = torch.autograd.grad(loss_clean, (model._gated_drug,
                                                        model._gated_disease),
                                           retain_graph=False)
            adv_delta = {'drug': args.adv_eps * g_d.sign(),
                         'disease': args.adv_eps * g_r.sign()}
            del loss_clean, g_d, g_r, dr_representation, train_score, \
                ce_loss, ssl_loss, train_prob, ap_loss
            torch.cuda.empty_cache()
            dr_representation, train_score, ssl_loss = model(
                drdr_graph, didi_graph, homo_graph_cpu, drug_feature, disease_feature,
                protein_feature, X_train, meta_paths_drug, meta_paths_disease,
                adj_matrices, labels=ssl_labels, adv_delta=adv_delta)
            ce_loss = cross_entropy(train_score, torch.flatten(Y_train))
            train_prob = fn.softmax(train_score, dim=-1)[:, 1]
            if getattr(args, 'use_ap', 1):
                ap_loss = soft_ap_loss(train_prob, torch.flatten(Y_train).float(),
                                       tau=ap_tau, neg_samples=dyn_neg,
                                       hard_ratio=ap_hard_ratio,
                                       sample=X_train, rank=ap_rank)
            else:
                ap_loss = torch.tensor(0.0, device=device)

        # 方向 F：梯度平衡（余弦一致性缩放 ap/ssl 权重）
        if args.use_grad_balance and epoch >= args.gb_warmup:
            g_ce = torch.autograd.grad(ce_loss, dr_representation,
                                       retain_graph=True)[0].detach()
            if ap_weight > 0:
                g_ap = torch.autograd.grad(ap_loss, dr_representation,
                                           retain_graph=True)[0].detach()
                cos_ap = fn.cosine_similarity(g_ce.flatten(), g_ap.flatten(), dim=0).clamp(0, 1)
                ap_weight = ap_weight * (0.5 + 0.5 * cos_ap)
            if getattr(args, 'use_ssl', 1) and ssl_weight > 0:
                g_ssl = torch.autograd.grad(ssl_loss, dr_representation,
                                            retain_graph=True)[0].detach()
                cos_ssl = fn.cosine_similarity(g_ce.flatten(), g_ssl.flatten(), dim=0).clamp(0, 1)
                ssl_weight = ssl_weight * (0.5 + 0.5 * cos_ssl)

        train_loss = ce_loss + ssl_weight * ssl_loss + ap_weight * ap_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
            print('ep %d/%d loss=%.4f' % (epoch + 1, args.epochs, train_loss.item()), flush=True)

    # ========== 最终预测：仅对 unknown 候选打分，取 Top-K ==========
    cand_prob = score_candidates(model).cpu().numpy()
    order = np.argsort(-cand_prob)
    prob_of = {cand_idx[int(p)]: float(cand_prob[p]) for p in range(len(cand_idx))}
    top_idx = [cand_idx[int(p)] for p in order[:args.topk]]

    drug_map = load_drug_id_and_name(args.data_dir, n_drugs)
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir,
                           'AD_keepKnown_top%d_%s_s%d.csv'
                           % (args.topk, args.dataset, args.random_seed))

    header = ['rank', 'drug_index', 'DrugBank_ID', 'drug_name',
              'P(positive)', 'known_AD_in_dataset']
    rows = []
    print('\n==== Top-%d predicted drugs for Alzheimer Disease (%s) [non-cold-start / '
          'known kept in train, unknown-only ranking] ====' % (args.topk, args.dataset))
    for r, i in enumerate(top_idx):
        db_id, name = drug_map[i]
        known = 'yes' if i in known_ad else 'no'
        rows.append([r + 1, i, db_id, name, '%.6f' % prob_of[i], known])
        print('%-4d %-10s %-14s P=%.6f  known=%s' % (r + 1, db_id, name, prob_of[i], known))

    pd.DataFrame(rows, columns=header).to_csv(out_csv, index=False)
    print('\nSaved ->', out_csv)
    print('Reminder: 本结果为非冷启动口径（AD 已知药参与训练、unknown 排序），'
          'P 值不可与全屏蔽冷启动口径直接比较；'
          '候选药物与 AD 的文献佐证（PMID）需在 PubMed 人工复核。')


if __name__ == '__main__':
    main()
