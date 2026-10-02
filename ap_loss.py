import torch
import numpy as np
import scipy.sparse as sp


def soft_ap_loss(scores, labels, tau=0.1, neg_samples=10000, hard_ratio=0.7,
                 sample=None, rank=2):
    """
    可微 Average Precision 代理损失（含 Hard Negative Mining）。

    对每个正样本计算 soft_rank ≈ 排名位置，精度 ≈ 1/soft_rank，
    损失 = 1 - mean(精度)。

    负采样策略：hard_ratio 比例从得分最高的负样本中取（hard negatives），
    其余随机采样保持多样性。

    方向 E（低秩难负样本）：当提供 sample（drug/disease 索引对）且 rank>0 时，
    对负样本在 drug×disease 得分矩阵上做低秩分解（S ≈ UV^T），用残差
    R = S - lowrank 排序难例——残差大的负样本属于"全局趋势之外仍被高估"的
    信息性难例（对应 MGTAL-DR 的低秩解耦难负采样）；SVD 失败时回退到按得分取 topk。

    Args:
        scores:      [N] 正类预测概率，shape (N,)
        labels:      [N] 0/1 标签，shape (N,)
        tau:         温度参数，控制 sigmoid 平滑度（越小越接近硬排名）
        neg_samples: 每次迭代采样的负样本总数
        hard_ratio:  难负样本占比，0.0=纯随机，1.0=纯难例
        sample:      [N, 2] 每个样本的 (drug_idx, disease_idx)，用于低秩分解
        rank:        低秩分解的秩，0 表示不使用低秩（直接按得分取 topk）

    Returns:
        scalar loss = 1.0 - AP
    """
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)

    pos_indices = pos_mask.nonzero(as_tuple=True)[0]
    neg_indices = neg_mask.nonzero(as_tuple=True)[0]

    if len(pos_indices) == 0:
        return torch.tensor(0.0, device=scores.device)

    n_neg = len(neg_indices)
    if n_neg == 0:
        return torch.tensor(0.0, device=scores.device)

    k = min(neg_samples, n_neg)

    # Hard Negative Mining: 取得分最高的 hard_ratio * k 个负样本
    # 这些负样本最有可能在排名中超过正样本（AUPR 瓶颈）
    n_hard = int(k * hard_ratio)
    n_random = k - n_hard

    neg_scores = scores[neg_indices]

    # 方向 E：低秩分解残差难例选择（失败时回退到按得分 topk）
    hard_top_idx = None
    if sample is not None and rank > 0 and n_hard > 0 and n_neg >= 4:
        try:
            drug_idx = sample[neg_indices, 0].cpu().numpy()
            dis_idx = sample[neg_indices, 1].cpu().numpy()
            n_d = int(drug_idx.max()) + 1
            n_di = int(dis_idx.max()) + 1
            r = min(rank, n_d - 1, n_di - 1)
            if r > 0 and n_d > r and n_di > r:
                neg_np = neg_scores.detach().cpu().numpy().astype(np.float64)
                M = sp.coo_matrix((neg_np, (drug_idx, dis_idx)), shape=(n_d, n_di)).tocsc()
                U, S, Vt = sp.linalg.svds(M, k=r)
                lowrank = (U * S) @ Vt
                resid = neg_np - lowrank[drug_idx, dis_idx]
                resid_t = torch.from_numpy(resid).to(scores.device)
                _, hard_top_idx = torch.topk(resid_t, min(n_hard, n_neg), largest=True)
        except Exception:
            hard_top_idx = None

    if hard_top_idx is None:
        _, hard_top_idx = torch.topk(neg_scores, min(n_hard, n_neg), largest=True)
    sampled_hard = neg_indices[hard_top_idx]

    # 随机采样剩余负样本，保证统计无偏和多样性
    if n_random > 0:
        # 排除已选中的 hard negatives 后随机采样
        hard_set = set(hard_top_idx.cpu().tolist())
        remaining_idx = torch.tensor(
            [i for i in range(n_neg) if i not in hard_set],
            device=scores.device
        )
        if len(remaining_idx) >= n_random:
            rand_perm = torch.randperm(len(remaining_idx), device=scores.device)[:n_random]
            sampled_random = neg_indices[remaining_idx[rand_perm]]
        else:
            sampled_random = neg_indices[remaining_idx]
    else:
        sampled_random = neg_indices[:0]

    sampled_neg = torch.cat([sampled_hard, sampled_random])

    # 合并正样本 + 采样负样本：[P + K]
    combined_indices = torch.cat([pos_indices, sampled_neg])
    combined_scores = scores[combined_indices]  # [P + K]

    # 分块计算 soft_rank，控制显存：diff 从 (P, P+K) 降为 (C, P+K)（C=chunk）。
    # 每个正样本的 soft_rank 都遍历全部 combined 样本，分块求和后取均值，
    # 与不分块的结果完全等价（OOM 修复：F 数据集 K 可 ramp 至 ~8 万，
    # 全量 diff ≈ 588MB，在 24GB 显存 + FGSM 双前向下会溢出）。
    chunk = 512
    total_precision = torch.tensor(0.0, device=scores.device)
    for start in range(0, len(pos_indices), chunk):
        chunk_pos = pos_indices[start:start + chunk]
        chunk_pos_scores = scores[chunk_pos]                 # [C]
        # 得分差矩阵: (C) × (P+K)
        diff = combined_scores.unsqueeze(0) - chunk_pos_scores.unsqueeze(1)  # [C, P+K]
        # soft_rank[i] = Σ_j σ((score_j - score_i) / tau)
        soft_rank = torch.sigmoid(diff / tau).sum(dim=1)     # [C]
        # precision_i = 1.0 / soft_rank_i
        precision = 1.0 / soft_rank.clamp(min=1e-8)          # [C]
        total_precision = total_precision + precision.sum()

    ap = total_precision / len(pos_indices)
    return 1.0 - ap
