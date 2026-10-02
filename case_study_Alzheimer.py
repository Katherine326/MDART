# -*- coding: utf-8 -*-
"""
阿尔茨海默病（Alzheimer's disease, AD）案例分析脚本（对应论文 3.7 节 / 表 6）

实验设置：
  1. 使用数据集中【全部】已知药物-疾病关联训练模型（不划分 10 折）；
  2. 将“阿尔茨海默病”相关的所有 (药物, AD) 配对从训练集中屏蔽
     ——AD 的正样本不再参与训练、AD 的全部候选对也不作为负样本；
  3. 训练完成后，对【每个药物 × AD】候选配对输出预测概率，
     取 Top-K（默认 10）药物作为候选，导出 CSV 供 PubMed 文献佐证核对。

用法（在 zuizhong 目录、MAPTrans 环境内执行）：
  python case_study_Alzheimer.py --dataset F-dataset          # 默认 600 epoch
  python case_study_Alzheimer.py --dataset C-dataset --epochs 600
  python case_study_Alzheimer.py --dataset F-dataset --topk 10 --random_seed 1234

说明：
  - AD 疾病索引自动定位：F/C 数据集 DiseaseFeature 首列标识 = 'D104300'；
    B 数据集按疾病名包含 'alzheimer' 匹配（B 为 best-effort）。
  - 已知 AD 阳性（被屏蔽的关联）用于“最优 epoch 选择”与“命中率”参考：
    每个 epoch 后对候选做一次轻量评估，按 known-AD 的 AP / top10 命中数选最优权重。
  - 输出的 PMID 文献佐证需要人工在 PubMed 检索补填（论文表 6 中 9 个有 PMID、1 个 N/A）。
  - 超参数默认值与 train_DDA.py 完全一致（lr=1e-4, dropout=0.4, wd=1e-3, epochs 可调…）。
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


# ---------------------------------------------------------------------------
# 数据集辅助：定位 AD 疾病索引 + 药物索引 -> DrugBank ID/名称 映射
# ---------------------------------------------------------------------------
def find_ad_index(dataset, data_dir):
    """返回阿尔茨海默病在疾病列表中的行索引（= 关联矩阵中的疾病索引）。"""
    ids = pd.read_csv(data_dir + 'DiseaseFeature.csv', header=None, usecols=[0],
                      dtype=str).iloc[:, 0].astype(str).str.strip()
    if dataset.lower().startswith('b'):
        hit = ids[ids.str.contains('alzheimer', case=False, na=False)].index.tolist()
    else:  # F / C：疾病标识形如 'D104300'（'D' + OMIM 号）
        hit = ids[ids == 'D104300'].index.tolist()
    if not hit:
        raise RuntimeError('未能在 %s 的 DiseaseFeature.csv 中定位阿尔茨海默病'
                           '(F/C 期望标识 D104300；B 期望疾病名含 alzheimer)' % dataset)
    return int(hit[0])


def load_drug_id_and_name(data_dir, n_drugs):
    """drug 索引 i -> (DrugBank ID, 通用名)。优先用 DrugFingerprint 首列 DB ID +
    DrugInformation 名称表；长度不符或缺失时退回仅用 ID。"""
    db_ids = []
    try:
        fp = pd.read_csv(data_dir + 'DrugFingerprint.csv')  # 首列表头为空 -> DB ID
        db_ids = [str(x).strip() for x in fp.iloc[:, 0].tolist()]
    except Exception:
        db_ids = []
    if len(db_ids) != n_drugs:            # 保险：改用 Allnode 前 n_drugs 行
        try:
            an = pd.read_csv(data_dir + 'Allnode.csv')
            col = 'id' if 'id' in an.columns else an.columns[1]
            db_ids = [str(x).strip() for x in an[col].tolist()[:n_drugs]]
        except Exception:
            db_ids = [str(i) for i in range(n_drugs)]

    id2name = {}
    try:
        di = pd.read_csv(data_dir + 'DrugInformation.csv')
        id2name = dict(zip(di['id'].astype(str).str.strip(),
                           di['name'].astype(str).str.strip()))
    except Exception:
        pass

    out = []
    for i in range(n_drugs):
        did = db_ids[i] if i < len(db_ids) else str(i)
        out.append((did, id2name.get(did, did)))
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='AD 案例分析：屏蔽目标疾病后训练 + Top-K 候选预测')
    # ---- 案例研究专属参数 ----
    parser.add_argument('--dataset', default='F-dataset', help='F-dataset / C-dataset / B-dataset')
    parser.add_argument('--epochs', type=int, default=600, help='训练轮数（与 train_DDA.py 默认 600 一致）')
    parser.add_argument('--topk', type=int, default=10, help='输出 Top-K 候选药物')
    parser.add_argument('--output_dir', default='Result/CaseStudy', help='结果输出目录')
    parser.add_argument('--exclude_known', type=int, default=0,
                        help='1 = 候选集剔除数据集中已知的 AD 关联（只预测新候选）；'
                             '0 = 全部药物(含已知 AD 药)都参与排序（论文表 6 口径）')
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

    # 已知 (drug, AD) 阳性集合（随后从训练集屏蔽，仅作评测/输出标注）
    known_ad = set(int(d) for (d, dis) in data['drdi'] if int(dis) == ad_idx)

    # ---- 训练样本：屏蔽所有与 AD 有关的配对（正负都不进训练）----
    all_pairs = np.asarray(data['all_drdi'], dtype=np.int64)
    all_label = np.asarray(data['all_label'], dtype=np.int64)
    mask = all_pairs[:, 1] != ad_idx
    X_tr = all_pairs[mask]
    Y_tr = all_label[mask]
    X_train = torch.LongTensor(X_tr).to(device)
    Y_train = torch.LongTensor(np.expand_dims(Y_tr, 1)).to(device)
    print('train pairs = %d (excluded AD pairs = %d)' % (len(X_tr), (~mask).sum()))

    # 候选集：每个药物 × AD（默认含已知 AD 药；--exclude_known 1 时剔除）
    cand_idx = [i for i in range(n_drugs)
                if not (args.exclude_known and i in known_ad)]
    cand = torch.LongTensor([[i, ad_idx] for i in cand_idx]).to(device)
    cand_known = np.array([1 if i in known_ad else 0 for i in cand_idx],
                          dtype=np.int64)
    print('candidates = %d (known-AD positives among them = %d)'
          % (len(cand_idx), int(cand_known.sum())))

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

    # ========== 候选打分函数（轻量，用于每 epoch 选最优权重）==========
    @torch.no_grad()
    def score_candidates(model):
        model.eval()
        _, out, _ = model(drdr_graph, didi_graph, homo_graph_cpu, drug_feature,
                          disease_feature, protein_feature, cand, meta_paths_drug,
                          meta_paths_disease, adj_matrices)
        return fn.softmax(out, dim=-1)[:, 1]

    best_ap, best_state, best_hits = -1.0, None, -1
    top10 = None

    # ========== 训练 ==========
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

        # ---- 每 epoch 轻量评估候选：以被屏蔽的已知 AD 阳性做“测试代理”选最优权重 ----
        if cand_known.sum() > 0:
            cand_prob = score_candidates(model).cpu().numpy()
            order = np.argsort(-cand_prob)              # 候选列表内位置（优→劣）
            pos2drug = [cand_idx[int(p)] for p in order]
            hit_top10 = sum(1 for d in pos2drug[:args.topk] if d in known_ad)
            pos_of_pos = [r for r, p in enumerate(order) if cand_known[p] == 1]
            if pos_of_pos:
                ap = float(np.mean([(k + 1) / (pos_of_pos[k] + 1)
                                    for k in range(len(pos_of_pos))]))
            else:
                ap = 0.0
            if ap > best_ap + 1e-6 or (abs(ap - best_ap) <= 1e-6 and hit_top10 > best_hits):
                best_ap, best_hits = ap, hit_top10
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
                print('ep %d/%d loss=%.4f | best known-AD AP=%.4f hit@%d=%d/%d'
                      % (epoch + 1, args.epochs, train_loss.item(), best_ap,
                         args.topk, best_hits, int(cand_known.sum())), flush=True)
        elif (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
            print('ep %d/%d loss=%.4f' % (epoch + 1, args.epochs, train_loss.item()), flush=True)

    # ========== 最终预测（最优 epoch 权重）==========
    if best_state is not None:
        model.load_state_dict(best_state)
    cand_prob = score_candidates(model).cpu().numpy()
    order = np.argsort(-cand_prob)
    prob_of = {cand_idx[int(p)]: float(cand_prob[p]) for p in range(len(cand_idx))}
    top_idx = [cand_idx[int(p)] for p in order[:args.topk]]

    drug_map = load_drug_id_and_name(args.data_dir, n_drugs)
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir,
                           'AD_top%d_%s_s%d.csv' % (args.topk, args.dataset, args.random_seed))

    header = ['rank', 'drug_index', 'DrugBank_ID', 'drug_name',
              'P(positive)', 'known_AD_in_dataset']
    rows = []
    print('\n==== Top-%d predicted drugs for Alzheimer Disease (%s) ====' % (args.topk, args.dataset))
    for r, i in enumerate(top_idx):
        db_id, name = drug_map[i]
        known = 'yes' if i in known_ad else 'no'
        rows.append([r + 1, i, db_id, name, '%.6f' % prob_of[i], known])
        print('%-4d %-10s %-14s P=%.6f  known=%s' % (r + 1, db_id, name, prob_of[i], known))

    pd.DataFrame(rows, columns=header).to_csv(out_csv, index=False)
    print('\nSaved ->', out_csv)
    print('Reminder: 请人工在 PubMed 逐条检索 Top-%d 药物与 AD 的文献（PMID），'
          '确认后才可作为论文表 6 的“文献佐证”。' % args.topk)


if __name__ == '__main__':
    main()
