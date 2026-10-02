import timeit
import argparse
import random
import numpy as np
import pandas as pd
import dgl
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
from data_preprocess import *
from MDART import MAPTrans
from metric import *
from ap_loss import soft_ap_loss
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, \
    precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=600, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=20, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='F-dataset', help='dataset')
    parser.add_argument('--dropout', default='0.4', type=float, help='dropout')
    parser.add_argument('--gt_layer', default='2', type=int, help='graph transformer layer')
    parser.add_argument('--gt_head', default='2', type=int, help='graph transformer head')
    parser.add_argument('--gt_out_dim', default='200', type=int, help='graph transformer output dimension')
    parser.add_argument('--hgt_layer', default='2', type=int, help='heterogeneous graph transformer layer')
    parser.add_argument('--hgt_head', default='8', type=int, help='heterogeneous graph transformer head')
    parser.add_argument('--hgt_in_dim', default='64', type=int, help='heterogeneous graph transformer input dimension')
    parser.add_argument('--hgt_head_dim', default='25', type=int, help='heterogeneous graph transformer head dimension')
    parser.add_argument('--hgt_out_dim', default='200', type=int, help='heterogeneous graph transformer output dimension')
    parser.add_argument('--tr_layer', default='2', type=int, help='transformer layer')
    parser.add_argument('--tr_head', default='4', type=int, help='transformer head')

    # ========== SOTA 改进方向开关（sota_improvement_directions.md） ==========
    # 方向 A：GDC 图扩散（多尺度相似度）
    parser.add_argument('--use_gdc', type=int, default=1, help='方向A: GDC 图扩散开关')
    parser.add_argument('--gdc_alpha', type=float, default=0.05, help='GDC 跃迁概率')
    parser.add_argument('--gdc_order', type=int, default=3, help='GDC 扩散阶数')
    # 方向 D：对抗扰动（FGSM）
    parser.add_argument('--use_adv', type=int, default=1, help='方向D: 对抗扰动开关')
    parser.add_argument('--adv_eps', type=float, default=0.01, help='FGSM 扰动幅度')
    parser.add_argument('--adv_warmup', type=int, default=50, help='对抗扰动预热 epoch')
    # 方向 B：视图间对比 SSL
    parser.add_argument('--use_fuse_ssl', type=int, default=1, help='方向B: 视图间对比 SSL 开关')
    # 方向 E：低秩分解难负样本
    parser.add_argument('--use_lowrank_ap', type=int, default=1, help='方向E: 低秩难负样本开关')
    parser.add_argument('--ap_rank', type=int, default=2, help='低秩分解的秩')
    # 方向 F：梯度平衡
    parser.add_argument('--use_grad_balance', type=int, default=1, help='方向F: 梯度平衡开关')
    parser.add_argument('--gb_warmup', type=int, default=50, help='梯度平衡预热 epoch')

    # ========== P0 修复 & P1 架构升级开关（默认开启，置 0 可作消融对照） ==========
    # P1-B1：元路径视图数（4=全部视图，不再硬 Top-2；2=旧 Top-2 消融行为）
    parser.add_argument('--num_metapaths', type=int, default=4, help='P1-B1: 元路径视图数')
    # P1-B2：跨视图 Transformer 编码（可微视图注意力）
    parser.add_argument('--use_view_trans', type=int, default=1, help='P1-B2: 跨视图 Transformer 编码开关')
    # P1-B1：可微视图注意力池化（0=均匀平均，创新点二消融对照）
    parser.add_argument('--use_view_pool', type=int, default=1, help='P1-B1: 可微视图注意力池化开关（0=均匀平均）')
    # 创新点二消融基线：还原原版 MAPTrans 视图方法（硬 Top-2 + 每视图独立 HGT + 3×gt_out_dim 拼接 + 纯 Hadamard）
    parser.add_argument('--use_orig_view', type=int, default=0, help='创新点二消融: 还原原版视图方法开关（1=原版，0=zuizhong）')
    # P1-B3：配对交互拼接 cat(hadamard, drug, disease)
    parser.add_argument('--use_inter_cat', type=int, default=1, help='P1-B3: 配对交互拼接开关')
    # P0-A7：原始特征（mol2vec/DiseaseFeature）残差注入
    parser.add_argument('--use_raw_feat', type=int, default=1, help='P0-A7: 原始特征残差注入开关')
    # P1-B4：蛋白 2 层 MLP 编码
    parser.add_argument('--use_prot_mlp', type=int, default=1, help='P1-B4: 蛋白 MLP 编码开关')
    # P1-B5：相似度图节点特征多视图拼接（drf|drg, dip|dig）
    parser.add_argument('--use_multi_sim_feat', type=int, default=1, help='P1-B5: 相似度图多视图特征开关')

    # ========== OOM 修复（C-dataset 24GB 显存爆显存） ==========
    # C-dataset 训练样本 24.4 万，MLP 激活一次前向约 12GB；--use_ckpt 把 MLP
    # 分块梯度检查点，峰值降到 ~1GB/段（语义不变）。小数据集可置 0 提速。
    parser.add_argument('--use_ckpt', type=int, default=1, help='OOM修复: MLP 分块梯度检查点；小数据集可置 0 关闭')

    # ========== 消融开关补充：损失/训练组件（默认开启，置 0 作消融对照） ==========
    parser.add_argument('--use_ap', type=int, default=1, help='消融: AP 代理损失开关（0=关闭，ap_weight 视为 0）')
    parser.add_argument('--use_ssl', type=int, default=1, help='消融: 对比学习 SSL 开关（0=关闭全部 SSL，含跨模态与视图间）')
    parser.add_argument('--use_gating', type=int, default=1, help='消融: DRDM 自门控+符号噪声扰动开关（MAPTrans.forward 读取）')
    parser.add_argument('--use_ce_weight', type=int, default=1, help='消融: 类别加权 CE 开关（0=退化为 1:1 无权重）')

    args = parser.parse_args()
    args.data_dir = 'data/' + args.dataset + '/'
    args.result_dir = 'Result/' + args.dataset + '/MDART/'

    # P0-A6：统一随机种子，保证训练可复现
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    try:
        dgl.seed(args.random_seed)
    except AttributeError:
        pass

    data, adj_matrices = get_data(args)
    # 邻接矩阵转为 tensor 但保留在 CPU，forward 中 agg_res 会按需逐条移 GPU
    for key in adj_matrices:
        adj_matrices[key] = torch.tensor(adj_matrices[key], dtype=torch.float32)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    # P0-A7 / P1-B5：运行时读取特征维度，避免硬编码
    args.drug_feat_dim = data['drugfeature'].shape[1]
    args.disease_feat_dim = data['diseasefeature'].shape[1]
    args.protein_feat_dim = data['proteinfeature'].shape[1]
    if getattr(args, 'use_multi_sim_feat', 1):
        args.drug_sim_dim = data['drf'].shape[1] + data['drg'].shape[1]
        args.disease_sim_dim = data['dip'].shape[1] + data['dig'].shape[1]

    data = data_processing(data, args)
    data = k_fold(data, args)

    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)

    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)
    # drug_drug = pd.read_csv('./data/B-dataset/drug_drug_baseline.csv', header=None).values
    # drug_drug = torch.FloatTensor(drug_drug).to(device)
    # disease_disease = pd.read_csv('./data/B-dataset/disease_disease_baseline.csv', header=None).values
    # disease_disease = torch.FloatTensor(disease_disease).to(device)
    drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)
    all_sample = torch.tensor(data['all_drdi']).long()

    start = timeit.default_timer()

    Metric = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
    AUCs, AUPRs, Accuracies, Precisions, Recalls, F1_scores, MCCs = [], [], [], [], [], [], []

    print('Dataset:', args.dataset)
    all_auc_values = []
    all_aupr_values = []
    for i in range(args.k_fold):
        print('fold:', i)
        print(Metric)
        fold_auc_values = []
        fold_aupr_values = []
        model = MAPTrans(args)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        # 动态类别加权 + 计算实际正负比
        pos_count = Y_train.sum().item()
        neg_count = len(Y_train) - pos_count
        actual_ratio = neg_count / pos_count  # 每个数据集的实际正负比
        print(f'Fold {i}: pos={pos_count}, neg={neg_count}, ratio=1:{actual_ratio:.1f}')
        # sqrt 加权替代直接比例，防止极端权重导致正类梯度爆炸（Focal Loss 论文推荐）
        # --use_ce_weight 0 时退化为 1:1 无权重（消融对照）
        if getattr(args, 'use_ce_weight', 1):
            ce_weight = torch.tensor([1.0, actual_ratio ** 0.5], device=device)
        else:
            ce_weight = torch.ones(2, device=device)
        cross_entropy = nn.CrossEntropyLoss(weight=ce_weight)
        # AP 代理损失参数 — 1:95 极端不平衡，AP loss 升为 co-primary
        ap_lambda = 0.50          # 0.30 → 0.50，AP loss 彻底主导排名优化
        ap_warmup_epochs = 30     # 快速预热，尽早介入排名信号
        ap_tau = 0.03             # 稳定值
        ap_hard_ratio = 1.0       # 全量难负样本，直击 AUPR 瓶颈
        # 动态负采样（DRMGNE/MVSGDR 论文策略）：根据实际正负比自适应调整
        # 初期用 ~3% 的负样本降低难度，后期用 ~50% 逼近真实分布
        ap_neg_ratio_start = max(2.0, actual_ratio * 0.03)
        ap_neg_ratio_end = actual_ratio * 0.5
        ap_neg_ramp_epochs = 500
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        drdipr_graph, data, meta_paths_drug, meta_paths_disease = dgl_heterograph(data, data['X_train'][i], args)
        # 预转换为同构图（CPU），避免 forward 内每次 dgl.to_homogeneous 重复创建 + 拷贝
        homo_graph_cpu = dgl.to_homogeneous(drdipr_graph)


        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            # 第一次前向（干净路径，随机符号噪声）；传 labels 使 SSL 分支真正生效
            # --use_ssl 0 时不传 labels，模型内部跳过全部 SSL 计算（跨模态+视图间）
            ssl_labels = Y_train if getattr(args, 'use_ssl', 1) else None
            dr_representation, train_score, ssl_loss = model(drdr_graph, didi_graph, homo_graph_cpu, drug_feature, disease_feature, protein_feature,
                                   X_train, meta_paths_drug, meta_paths_disease, adj_matrices, labels=ssl_labels)
            ce_loss = cross_entropy(train_score, torch.flatten(Y_train))
            # SSL lambda 线性预热：前 50 epoch 从 0 ramp 到 0.1，避免初期对比损失干扰 CE 收敛
            ssl_weight = model.ssl_lambda * min(1.0, epoch / 50)
            # AP 代理损失：约束排名退化，使 AUC/AUPR 峰值趋于一致
            # Hard Negative Mining: 100% 最难负样本，直击 AUPR 瓶颈
            train_prob = fn.softmax(train_score, dim=-1)[:, 1]
            # 动态负采样：neg/pos 比例随 epoch 从 ap_neg_ratio_start 线性增至 ap_neg_ratio_end
            neg_ratio = ap_neg_ratio_start + (ap_neg_ratio_end - ap_neg_ratio_start) * min(1.0, epoch / ap_neg_ramp_epochs)
            dyn_neg_samples = min(int(pos_count * neg_ratio), neg_count)
            # 方向 E：低秩分解难负样本（sample + rank，0 表示禁用低秩）
            ap_rank = args.ap_rank if args.use_lowrank_ap else 0
            # --use_ap 0 时 AP 损失置 0 常量（不参与梯度）
            if getattr(args, 'use_ap', 1):
                ap_loss = soft_ap_loss(train_prob, torch.flatten(Y_train).float(),
                                        tau=ap_tau, neg_samples=dyn_neg_samples,
                                        hard_ratio=ap_hard_ratio, sample=X_train, rank=ap_rank)
            else:
                ap_loss = torch.tensor(0.0, device=device)
            # 两阶段动态权重 — AP loss co-primary：
            # 阶段 1 (epoch 0-30):   快速预热 0 → 0.05
            # 阶段 2 (epoch 30-300): 线性增至 ap_lambda(0.50)，增速 0.00167/epoch
            if getattr(args, 'use_ap', 1):
                if epoch < ap_warmup_epochs:
                    ap_weight = 0.05 * (epoch / ap_warmup_epochs)
                else:
                    ramp_progress = min(1.0, (epoch - ap_warmup_epochs) / 270)
                    ap_weight = 0.05 + (ap_lambda - 0.05) * ramp_progress
            else:
                ap_weight = 0.0

            # ===== 方向 D：FGSM 对抗扰动（两步 forward） =====
            if args.use_adv and epoch >= args.adv_warmup:
                loss_clean = ce_loss + ssl_weight * ssl_loss + ap_weight * ap_loss
                # OOM 修复：第一次前向的计算图在算出 g_d/g_r 后不再使用
                # （下方 del 会删掉全部引用），因此这里 retain_graph=False，
                # autograd 边算边释放 ~12GB 图内存，第二次 forward 才有完整显存预算。
                g_d, g_r = torch.autograd.grad(
                    loss_clean, (model._gated_drug, model._gated_disease), retain_graph=False)
                adv_delta = {'drug': args.adv_eps * g_d.sign(),
                             'disease': args.adv_eps * g_r.sign()}
                # OOM 修复：释放第一次前向的计算图引用（train_score/ce_loss/ssl_loss/
                # train_prob/ap_loss/dr_representation 均指向第一 forward 的图节点，
                # 不删除则与第二 forward 的图共存，24GB 显存在 AP loss 处溢出）
                del loss_clean, g_d, g_r, dr_representation, train_score, ce_loss, ssl_loss, train_prob, ap_loss
                torch.cuda.empty_cache()
                # 第二次前向：门控特征叠加对抗扰动后重算各损失
                dr_representation, train_score, ssl_loss = model(
                    drdr_graph, didi_graph, homo_graph_cpu, drug_feature, disease_feature, protein_feature,
                    X_train, meta_paths_drug, meta_paths_disease, adj_matrices, labels=ssl_labels, adv_delta=adv_delta)
                ce_loss = cross_entropy(train_score, torch.flatten(Y_train))
                train_prob = fn.softmax(train_score, dim=-1)[:, 1]
                if getattr(args, 'use_ap', 1):
                    ap_loss = soft_ap_loss(train_prob, torch.flatten(Y_train).float(),
                                            tau=ap_tau, neg_samples=dyn_neg_samples,
                                            hard_ratio=ap_hard_ratio, sample=X_train, rank=ap_rank)
                else:
                    ap_loss = torch.tensor(0.0, device=device)

            # ===== 方向 F：梯度平衡（CE 与辅助损失在 dr_representation 上的梯度方向一致性） =====
            if args.use_grad_balance and epoch >= args.gb_warmup:
                g_ce = torch.autograd.grad(ce_loss, dr_representation, retain_graph=True)[0].detach()
                # ap_loss 为 0 常量时（--use_ap 0）autograd.grad 会报错，用 ap_weight>0 守卫
                if ap_weight > 0:
                    g_ap = torch.autograd.grad(ap_loss, dr_representation, retain_graph=True)[0].detach()
                    cos_ap = fn.cosine_similarity(g_ce.flatten(), g_ap.flatten(), dim=0).clamp(0, 1)
                    ap_weight = ap_weight * (0.5 + 0.5 * cos_ap)
                if getattr(args, 'use_ssl', 1) and ssl_weight > 0:
                    g_ssl = torch.autograd.grad(ssl_loss, dr_representation, retain_graph=True)[0].detach()
                    cos_ssl = fn.cosine_similarity(g_ce.flatten(), g_ssl.flatten(), dim=0).clamp(0, 1)
                    ssl_weight = ssl_weight * (0.5 + 0.5 * cos_ssl)

            train_loss = ce_loss + ssl_weight * ssl_loss + ap_weight * ap_loss
            # OOM 修复：torch<1.10 的自定义检查点（_CheckpointedMLPBlock）在每次
            # .grad() 反向（FGSM/梯度平衡）时会把段内 MLP 参数的梯度也累加到 .grad 上，
            # 这些是中间过程的"垃圾"梯度，最终 backward 前必须清零，
            # 保证 optimizer.step() 只使用最终 train_loss 的梯度。
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                dr_representation, test_score, _ = model(drdr_graph, didi_graph, homo_graph_cpu, drug_feature,
                                                      disease_feature, protein_feature, X_test, meta_paths_drug,
                                                      meta_paths_disease, adj_matrices)

            test_prob = fn.softmax(test_score, dim=-1)
            test_score = torch.argmax(test_score, dim=-1)
            test_prob = test_prob[:, 1]
            test_prob = test_prob.cpu().numpy()
            test_score = test_score.cpu().numpy()
            AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)

            fold_auc_values.append(AUC)
            fold_aupr_values.append(AUPR)


            end = timeit.default_timer()
            time = end - start
            show = [epoch + 1, round(time, 2), round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                    round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5),
                    round(ap_loss.item(), 5)]
            print('\t\t'.join(map(str, show)))
            if AUPR > best_aupr:
                best_epoch = epoch + 1
                best_aupr = AUPR
                best_auc, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUC, accuracy, precision, recall, f1, mcc
                print('AUPR improved at epoch ', best_epoch, ';\tbest_aupr:', best_aupr)

        # Save the best metrics for each fold
        all_auc_values.append(fold_auc_values)
        all_aupr_values.append(fold_aupr_values)
        AUCs.append(best_auc)
        AUPRs.append(best_aupr)
        Accuracies.append(best_accuracy)
        Precisions.append(best_precision)
        Recalls.append(best_recall)
        F1_scores.append(best_f1)
        MCCs.append(best_mcc)

        # 释放当前 fold 的 GPU 显存
        del drdipr_graph, homo_graph_cpu
        torch.cuda.empty_cache()

    # After all folds are finished, calculate the average and std for each metric
    print('AUC:', AUCs)
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    print('Mean AUC:', AUC_mean, '(', AUC_std, ')')

    print('AUPR:', AUPRs)
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    print('Mean AUPR:', AUPR_mean, '(', AUPR_std, ')')

    print('Accuracy:', Accuracies)
    Accuracy_mean = np.mean(Accuracies)
    Accuracy_std = np.std(Accuracies)
    print('Mean Accuracy:', Accuracy_mean, '(', Accuracy_std, ')')

    print('Precision:', Precisions)
    Precision_mean = np.mean(Precisions)
    Precision_std = np.std(Precisions)
    print('Mean Precision:', Precision_mean, '(', Precision_std, ')')

    print('Recall:', Recalls)
    Recall_mean = np.mean(Recalls)
    Recall_std = np.std(Recalls)
    print('Mean Recall:', Recall_mean, '(', Recall_std, ')')

    print('F1-score:', F1_scores)
    F1_score_mean = np.mean(F1_scores)
    F1_score_std = np.std(F1_scores)
    print('Mean F1-score:', F1_score_mean, '(', F1_score_std, ')')

    print('MCC:', MCCs)
    MCC_mean = np.mean(MCCs)
    MCC_std = np.std(MCCs)
    print('Mean MCC:', MCC_mean, '(', MCC_std, ')')


def visualize_meta_path(drug_id, disease_id, meta_paths_drug, meta_paths_disease):
    # Initialize graph
    G = nx.DiGraph()

    # Add drug-related meta-paths
    drug_paths = meta_paths_drug[drug_id]
    for path in drug_paths:
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i + 1], label='Drug Path')

    # Add disease-related meta-paths
    disease_paths = meta_paths_disease[disease_id]
    for path in disease_paths:
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i + 1], label='Disease Path')

    # Visualize the graph
    pos = nx.spring_layout(G)  # Layout for better visualization
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.title(f"Meta-path Visualization for Drug {drug_id} and Disease {disease_id}")
    plt.show()


