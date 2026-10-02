import dgl.nn.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, cosine_similarity

from model import gt_net_drug, gt_net_disease
from torch.nn import Parameter
import  warnings
import torch.nn.functional as F
import os
import torch
import pandas as pd
import inspect
from torch.nn.functional import cosine_similarity
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========== OOM 修复：torch.utils.checkpoint 兼容导入 ==========
# 旧版 torch 的 torch.utils.checkpoint 不会被 import torch 自动导入，
# 直接 torch.utils.checkpoint.checkpoint(...) 会报
#   AttributeError: module 'torch.utils' has no attribute 'checkpoint'
# 因此这里显式 import。且 use_reentrant 参数是 torch>=1.10 才有的：
#   - torch >= 1.10：用官方非 reentrant 检查点（与 .grad() 兼容）
#   - torch <  1.10：官方 reentrant 检查点与 torch.autograd.grad() 不兼容
#     （RuntimeError: Checkpointing is not compatible with .grad()...），
#     改用下方自定义 _CheckpointedMLPBlock（同样的"前向只存输入、反向重算"语义）。
try:
    from torch.utils.checkpoint import checkpoint as _ckpt
    _CKPT_HAS_REENTRANT = 'use_reentrant' in inspect.signature(_ckpt).parameters
except (ImportError, AttributeError):
    _ckpt = None
    _CKPT_HAS_REENTRANT = False


class _CheckpointedMLPBlock(torch.autograd.Function):
    """自定义分块梯度检查点（torch<1.10 时替代官方 reentrant checkpoint）。

    官方 reentrant checkpoint 内部用 torch.autograd.backward 重算，与
    torch.autograd.grad() 不兼容（训练里 FGSM/梯度平衡都依赖 .grad()）。
    本实现：forward 在 no_grad 下前向，只保存输入与 RNG 状态（不保存激活）；
    backward 恢复 RNG 后重算段内前向，用 autograd.grad 求 (输入, 段内参数)
    的梯度，参数梯度手动累加到 p.grad（段内参数被封装在 Function 内，
    引擎接触不到，必须自己累加）。
    """
    @staticmethod
    def forward(ctx, block, x):
        ctx.block = block
        ctx.rng_state = torch.get_rng_state()
        ctx.cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        ctx.save_for_backward(x)
        with torch.no_grad():
            return block(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        torch.set_rng_state(ctx.rng_state)
        if ctx.cuda_rng_state is not None:
            torch.cuda.set_rng_state(ctx.cuda_rng_state)
        with torch.enable_grad():
            out = ctx.block(x)
        params = [p for p in ctx.block.parameters() if p.requires_grad]
        grads = torch.autograd.grad(out, (x,) + tuple(params), grad_out, allow_unused=True)
        x_grad = grads[0]
        for p, g in zip(params, grads[1:]):
            if g is not None:
                p.grad = g if p.grad is None else p.grad + g
        return None, x_grad

class SelfAttentionLayer(torch.nn.Module):
    """标准自注意力精炼层：Q/K/V 注意力 + output 投影 + 残差。

    P0-A2 修复：原实现 forward 中未使用 self.output，该线性层参数永不参与学习；
    此处补全 output 投影并加残差连接（与输入同维）。
    """
    def __init__(self, input_dim, output_dim):
        super(SelfAttentionLayer, self).__init__()
        self.query = torch.nn.Linear(input_dim, output_dim)
        self.key = torch.nn.Linear(input_dim, output_dim)
        self.value = torch.nn.Linear(input_dim, output_dim)
        self.output = torch.nn.Linear(output_dim, input_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention_scores = torch.matmul(q, k.transpose(0, 1)) / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        # P0-A2：output 投影 + 残差
        out = x + self.output(attention_output)
        return out

def compute_attention_scores(views):
    """视图间余弦相似度打分（保留用于 num_metapaths<4 时的 Top-k 消融路径）。"""
    attention_scores = []

    # 计算每个视图的注意力分数
    for i in range(len(views)):
        attention_weight = 0
        for j in range(len(views)):
            if i != j:
                sim = cosine_similarity(views[i], views[j], dim=1)
                attention_weight += torch.sum(sim).item()

        # 添加当前视图的注意力得分
        attention_scores.append(attention_weight)

    # 转为 Tensor 并归一化所有视图的注意力得分
    attention_scores = torch.tensor(attention_scores, dtype=torch.float32)
    attention_scores = attention_scores / torch.sum(attention_scores)
    return attention_scores


def aggres(feature, metapaths, adj_matrices, drug, disease, protein, attention_layer, top_k=None):
    """元路径聚合。

    P0-A1 修复：attention_layer 由模型注册后传入（参数参与训练），
              不再在函数内每次 forward 新建随机初始化的层。
    P0-A2 修复：基础特征只精炼一次（base_feature），每条元路径从同一份
              特征出发，消除原实现"循环内原地覆盖 feature"导致的路径顺序依赖。
    P1-B1：top_k=None 返回全部视图；top_k<N 时按原余弦打分取 Top-k（消融）。
    """
    aggregated_features = []
    base_feature = attention_layer(feature).to(feature.device)
    for metapath, path in metapaths.items():
        adj_product = None
        node_type_feature = None
        for edge in path:
            node_type, edge_type, _ = edge
            adj_matrix = adj_matrices[edge_type].to(feature.device)  # 已预转为 tensor，仅移 device
            if adj_product is None:
                adj_product = adj_matrix
            else:
                if node_type == "drug":
                    node_type_feature = drug
                elif node_type == "disease":
                    node_type_feature = disease
                elif node_type == "protein":
                    node_type_feature = protein
                adj_node_type_feature = torch.matmul(adj_product, node_type_feature)
                adj_product = torch.matmul(adj_node_type_feature, adj_matrix)
        current_features_drug = torch.matmul(adj_product, base_feature)
        current_features = F.normalize(current_features_drug, p=2, dim=1)
        aggregated_features.append(current_features)

    if top_k is not None and 0 < top_k < len(aggregated_features):
        # 消融路径：保持旧的"余弦打分 Top-k"行为
        attention_scores = compute_attention_scores(aggregated_features)
        _, top_indices = torch.topk(attention_scores, top_k, largest=True)
        top_views = [aggregated_features[i] for i in top_indices]
        return top_views
    return aggregated_features


class MAPTrans(nn.Module):
    def __init__(self, args):
        super(MAPTrans, self).__init__()
        self.args = args
        self.drug_linear = nn.Linear(200, args.hgt_in_dim)
        self.disease_linear = nn.Linear(200, args.hgt_in_dim)
        # P1-B4：蛋白分支 2 层 MLP（--use_prot_mlp=1 时启用），单层 linear 保留作消融
        protein_in = getattr(args, 'protein_feat_dim', 320)
        self.protein_linear = nn.Linear(protein_in, args.hgt_in_dim)
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_in, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, args.hgt_in_dim),
            nn.LayerNorm(args.hgt_in_dim),
        )
        # P0-A7：原始特征（mol2vec 300d / DiseaseFeature 64d）此前被 forward 覆盖
        #        从未使用，此处新增投影层做残差注入（--use_raw_feat 开关）
        self.drug_raw_proj = nn.Linear(getattr(args, 'drug_feat_dim', 300), args.hgt_in_dim)
        self.disease_raw_proj = nn.Linear(getattr(args, 'disease_feat_dim', 64), args.hgt_in_dim)

        self.protein = nn.Linear(args.hgt_in_dim, args.protein_number)
        self.disease = nn.Linear(args.hgt_in_dim, args.disease_number)
        self.drug = nn.Linear(args.hgt_in_dim, args.drug_number)
        
        # ========== DRDM Debiasing: Self-Gating ==========
        # bias 初始化为正值，让 sigmoid 接近 1（门初始"全开"），后续自适应关闭噪声特征
        self.gating_weightd_b = nn.Parameter(torch.full((1, args.hgt_in_dim), 2.0))
        self.gating_weightd = nn.Parameter(torch.FloatTensor(args.hgt_in_dim, args.hgt_in_dim))
        nn.init.xavier_normal_(self.gating_weightd.data)
        self.gating_weightr_b = nn.Parameter(torch.full((1, args.hgt_in_dim), 2.0))
        self.gating_weightr = nn.Parameter(torch.FloatTensor(args.hgt_in_dim, args.hgt_in_dim))
        nn.init.xavier_normal_(self.gating_weightr.data)

        # ========== Contrastive Learning Temperature ==========
        self.ssl_temp = 0.1  # 上调温度减少噪声梯度
        self.ssl_lambda = 0.1

        # P1-B1：元路径视图数（默认 4 = 全部视图；2 = 旧的 Top-2 消融行为）
        # 元路径在 data_preprocess 中硬编码为 4 条，这里 clamp 到 [1, 4]
        # --use_orig_view 1 时强制 =2（原版：余弦打分硬 Top-2 选 2 个视图）
        if getattr(args, 'use_orig_view', 0):
            self.num_metapaths = 2
        else:
            self.num_metapaths = max(1, min(getattr(args, 'num_metapaths', 4), 4))
        # P0-A1：元路径聚合注意力层注册为模型模块（参数参与训练）
        self.attn_drug = SelfAttentionLayer(args.hgt_in_dim, args.hgt_in_dim)
        self.attn_disease = SelfAttentionLayer(args.hgt_in_dim, args.hgt_in_dim)

        # P1-B1/B2：融合视图维度。实测 4 视图 × HGT 在 24GB 显存仍 OOM（HGT 显存
        # 随视图数线性增长）；改为"语义视图先池化 → HGT 只跑 1 次 → 与相似度视图拼接"：
        #   - B1 可微视图注意力：对 4 条元路径语义视图 softmax 加权求和（HGT 之前）
        #   - B2 跨视图 Transformer：在 4 视图序列上做自注意力（d_model=64）
        #   - 融合 = cat(HGT输出, 相似度视图) → 2×gt_out_dim
        # --use_orig_view 1 时还原原版：2 视图各自独立过 HGT → 拼接 3×gt_out_dim → 纯 Hadamard
        if getattr(args, 'use_orig_view', 0):
            self.fuse_dim = 3 * args.gt_out_dim
            self.inter_dim = self.fuse_dim
        else:
            self.fuse_dim = 2 * args.gt_out_dim
            # P1-B3：配对交互维度（拼接 hadamard+drug+disease 时 ×3）
            self.inter_dim = 3 * self.fuse_dim if getattr(args, 'use_inter_cat', 1) else self.fuse_dim
        # P1-B1：语义视图注意力打分（对元路径视图序列加权，作用在 HGT 之前）
        self.view_attn_sem_drug = nn.Linear(args.hgt_in_dim, 1)
        self.view_attn_sem_disease = nn.Linear(args.hgt_in_dim, 1)

        # SSL 投影头：解耦对比学习空间与分类表示空间（SimCLR 标准做法）
        self.ssl_projector = nn.Sequential(
            nn.Linear(self.fuse_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        # 方向 B：相似度视图投影头，用于"语义视图 ↔ 相似度视图"视图间对比
        self.ssl_projector_fuse = nn.Sequential(
            nn.Linear(args.gt_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # P1-B5：相似度图 GT 的节点特征维度（multi_sim_feat 时为拼接特征宽度）
        drug_node_dim = getattr(args, 'drug_sim_dim', args.drug_number)
        disease_node_dim = getattr(args, 'disease_sim_dim', args.disease_number)
        self.gt_drug = gt_net_drug.GraphTransformer(device, args.gt_layer, drug_node_dim, args.gt_out_dim, args.gt_out_dim,
                                                    args.gt_head, args.dropout)
        self.gt_disease = gt_net_disease.GraphTransformer(device, args.gt_layer, disease_node_dim, args.gt_out_dim,
                                                    args.gt_out_dim, args.gt_head, args.dropout)

        self.hgt_dgl = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, int(args.hgt_in_dim/args.hgt_head), args.hgt_head, 3, 3, args.dropout)
        self.hgt_dgl_last = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, args.hgt_head_dim, args.hgt_head, 3, 3, args.dropout)
        self.hgt = nn.ModuleList()
        for l in range(args.hgt_layer-1):
            self.hgt.append(self.hgt_dgl)
        self.hgt.append(self.hgt_dgl_last)

        # P1-B2：跨视图 Transformer 编码器 —— 在 4 条元路径语义视图序列上做
        # 自注意力（可微视图间注意力），替代旧的硬 Top-2（--use_view_trans 开关）。
        # 作用于 HGT 之前的 64 维语义视图（d_model=hgt_in_dim）。
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.hgt_in_dim, nhead=args.tr_head)
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
        self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

        self.mlp = nn.Sequential(
            nn.Linear(self.inter_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )
        # ========== OOM 修复：MLP 分块梯度检查点（--use_ckpt） ==========
        # C-dataset 训练样本 24.4 万，MLP 一次前向的激活约 12GB，24GB 显存直接 OOM。
        # 按 (Linear+ReLU+Dropout) 分 4 段 checkpoint：前向只保存每段输入，
        # 反向时逐段重算激活，峰值从 ~12GB 降到 ~1GB/段。训练语义完全不变。
        # 小数据集（F/B 等装得下）可 --use_ckpt 0 关闭，省去重算开销。
        self.use_ckpt = bool(getattr(args, 'use_ckpt', 1))

        self.in_dim = 64
        self.hidden_dim = 64
        self.dropout = 0.4
        self.feature_node_name = ['drug', 'disease']

        # 跨模态语义聚合投影（按元路径视图数扩展）
        self.drug_disease = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.disease_number, (args.disease_number + 153) // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear((args.disease_number + 153) // 2, (args.disease_number + 153) // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear((args.disease_number + 153) // 2, 64),
            )
            for _ in range(self.num_metapaths)
        ])
        self.disease_drug = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.drug_number, (args.drug_number + 332) // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear((args.drug_number + 332) // 2, (args.drug_number + 332) // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear((args.drug_number + 332) // 2, 64),
            )
            for _ in range(self.num_metapaths)
        ])

        # transformer for feature semantic aggregation
        self.Q_drug = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.K_drug = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.V_drug = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.Q_disease = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.K_disease = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.V_disease = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.weights_list = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim)) for _ in
                                       range(self.num_metapaths)])

        self.beta_drug = Parameter(torch.ones(1))
        self.beta_disease = Parameter(torch.ones(1))

    # ========== DRDM Debiasing: Self-Gating ==========
    def self_gating_disease(self, em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weightd) + self.gating_weightd_b))

    def self_gating_drug(self, em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weightr) + self.gating_weightr_b))

    # ========== DRDM Debiasing: Noise Perturbation ==========
    def perturb_embedding(self, emb):
        """Add sign-directed Gaussian noise during training for robustness."""
        if self.training:
            sign = torch.sign(emb)
            random_noise = F.normalize(torch.randn_like(emb).to(emb.device)) * 0.005
            emb = emb + sign * random_noise
        return emb

    # ========== Graph Contrastive Learning Loss ==========
    def ssl_loss_views(self, view1, view2):
        """InfoNCE contrastive loss between two views of the same nodes."""
        if view1.shape[0] == 0:
            return torch.tensor(0.0).to(view1.device)
        norm_v1 = F.normalize(view1, p=2, dim=1)
        norm_v2 = F.normalize(view2, p=2, dim=1)
        pos_score = torch.sum(torch.mul(norm_v1, norm_v2), dim=1)
        # Use chunked computation to avoid O(N^2) memory
        N = norm_v1.shape[0]
        chunk_size = 4096
        all_score_sum = torch.zeros(N, device=norm_v1.device)
        for i in range(0, N, chunk_size):
            chunk = norm_v1[i:i + chunk_size]
            scores = torch.mm(chunk, norm_v2.T) / self.ssl_temp
            all_score_sum[i:i + chunk_size] = torch.sum(torch.exp(scores), dim=1)
        pos_score = torch.exp(pos_score / self.ssl_temp)
        ssl_loss = -torch.sum(torch.log(pos_score / (all_score_sum + 1e-8))) / N
        return ssl_loss

    def ssl_loss_cross(self, drug_emb, disease_emb, sample, labels):
        """Cross-modal contrastive loss between drug-disease pairs.
        Uses stratified sampling to ensure positive pair ratio ≥ 1:2."""
        drug_idx, disease_idx = sample[:, 0], sample[:, 1]
        N_all = drug_idx.shape[0]
        if N_all == 0:
            return torch.tensor(0.0).to(drug_emb.device)

        # 分层采样：保证正负比 ≥ 1:2，避免对比信号被假阴性淹没
        pos_mask = labels.flatten() == 1
        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(~pos_mask)[0]
        pos_sample = min(2048, len(pos_idx))
        neg_sample = min(4096, len(neg_idx))
        perm_pos = pos_idx[torch.randperm(len(pos_idx), device=drug_emb.device)[:pos_sample]]
        perm_neg = neg_idx[torch.randperm(len(neg_idx), device=drug_emb.device)[:neg_sample]]
        selected = torch.cat([perm_pos, perm_neg])
        N = len(selected)

        drug_batch = drug_emb[drug_idx[selected]]
        disease_batch = disease_emb[disease_idx[selected]]
        norm_drug = F.normalize(drug_batch, p=2, dim=1)
        norm_disease = F.normalize(disease_batch, p=2, dim=1)
        pos_score = torch.sum(torch.mul(norm_drug, norm_disease), dim=1)
        # Chunked denominator computation to avoid materializing N×N matrix
        chunk_size = 4096
        all_score_sum = torch.zeros(N, device=norm_drug.device)
        for i in range(0, N, chunk_size):
            chunk = norm_drug[i:i + chunk_size]
            scores = torch.mm(chunk, norm_disease.T) / self.ssl_temp
            all_score_sum[i:i + chunk_size] = torch.sum(torch.exp(scores), dim=1)
        pos_score = torch.exp(pos_score / self.ssl_temp)
        ssl_loss = -torch.sum(torch.log(pos_score / (all_score_sum + 1e-8))) / N
        return ssl_loss

    def _score_mlp(self, x):
        """分块梯度检查点的 MLP 打分前向（24GB 显存 OOM 修复）。

        训练模式下把 self.mlp 切成 4 段 (Linear+ReLU+Dropout)，每段检查点化：
        前向不保存段内激活，反向按需重算，把 C-dataset（24.4 万样本）
        MLP 的激活峰值从 ~12GB 降到 ~1GB/段。
        eval / --use_ckpt=0 / torch 无 checkpoint 时直接走原始 self.mlp。
        """
        if _ckpt is None or not self.use_ckpt or not self.training:
            return self.mlp(x)
        blocks = (self.mlp[0:3], self.mlp[3:6], self.mlp[6:9], self.mlp[9:])
        for block in blocks:
            if _CKPT_HAS_REENTRANT:
                x = _ckpt(block, x, use_reentrant=False)
            else:
                # torch < 1.10：官方 reentrant checkpoint 与 .grad() 不兼容，
                # 用自定义实现（语义一致，且兼容 .grad()）
                x = _CheckpointedMLPBlock.apply(block, x)
        return x

    def forward(self, drdr_graph, didi_graph, homo_graph_cpu, drug_feature, disease_feature, protein_feature, sample, meta_paths_drug, meta_paths_disease, adj_matrices, labels=None, adv_delta=None):
        # P0-A7：保留原始特征副本（原实现中 drug_feature/disease_feature 入参
        #        被后续赋值直接覆盖，mol2vec/DiseaseFeature 实际从未被模型使用）
        drug_feature_raw = drug_feature
        disease_feature_raw = disease_feature
        protein_feature_raw = protein_feature

        dr_sim = self.gt_drug(drdr_graph)
        di_sim = self.gt_disease(didi_graph)

        # P0-A3 修复：疾病使用独立的 disease_linear（原误用 drug_linear 共享权重）
        drug_feature = self.drug_linear(dr_sim)
        disease_feature = self.disease_linear(di_sim)
        # P1-B4：蛋白分支 2 层 MLP（开关关闭时退回单层线性）
        if getattr(self.args, 'use_prot_mlp', 1):
            protein_feature_64 = self.protein_encoder(protein_feature_raw)
        else:
            protein_feature_64 = self.protein_linear(protein_feature_raw)

        # P0-A7：原始分子/疾病语义特征残差注入（--use_raw_feat=0 关闭，退回旧行为）
        if getattr(self.args, 'use_raw_feat', 1):
            drug_feature = drug_feature + self.drug_raw_proj(drug_feature_raw)
            disease_feature = disease_feature + self.disease_raw_proj(disease_feature_raw)

        # ========== DRDM Debiasing: Self-Gating ==========
        # --use_gating 0 时旁路自门控（消融对照），特征保持未门控状态
        if getattr(self.args, 'use_gating', 1):
            drug_feature_gated = self.self_gating_drug(drug_feature)
            disease_feature_gated = self.self_gating_disease(disease_feature)
        else:
            drug_feature_gated = drug_feature
            disease_feature_gated = disease_feature

        # ========== DRDM Debiasing: Noise Perturbation ==========
        # 方向 D：提供 adv_delta 时使用基于梯度的 FGSM 对抗扰动，否则用随机符号噪声
        # （--use_gating 0 时连符号噪声也不加；FGSM 扰动由 --use_adv 独立控制）
        if adv_delta is not None:
            drug_feature_gated = drug_feature_gated + adv_delta['drug']
            disease_feature_gated = disease_feature_gated + adv_delta['disease']
        elif getattr(self.args, 'use_gating', 1):
            drug_feature_gated = self.perturb_embedding(drug_feature_gated)
            disease_feature_gated = self.perturb_embedding(disease_feature_gated)
        # 缓存门控特征，供训练侧基于梯度的对抗扰动（方向 D）取梯度
        self._gated_drug = drug_feature_gated
        self._gated_disease = disease_feature_gated

        drug = self.drug(drug_feature)
        disease = self.disease(disease_feature)
        protein = self.protein(protein_feature_64)

        # P0-A1/A2 + P1-B1：聚合注意力层由模型注册后传入；视图数可配
        # （num_metapaths>=4 时返回全部视图，不再硬 Top-2 丢弃信息）
        top_k = None if self.num_metapaths >= 4 else self.num_metapaths
        aggregated_features_drug = aggres(drug_feature_gated, meta_paths_drug, adj_matrices, drug, disease, protein,
                                          self.attn_drug, top_k=top_k)
        aggregated_features_disease = aggres(disease_feature_gated, meta_paths_disease, adj_matrices, drug, disease, protein,
                                             self.attn_disease, top_k=top_k)
        dr_trans = torch.stack(aggregated_features_drug)
        di_trans = torch.stack(aggregated_features_disease)
        drug_proj = dr_trans.transpose(0, 1)
        disease_proj = di_trans.transpose(0, 1)

        # 3. Transformer-based Semantic Aggregation
        Q_drug = self.Q_drug(drug_proj)
        K_drug = self.K_drug(drug_proj)
        V_drug = self.V_drug(drug_proj)
        Q_drug1 = Q_drug.transpose(0, 1)
        K_drug1 = K_drug.transpose(0, 1)
        Q_disease = self.Q_disease(disease_proj)
        K_disease = self.K_disease(disease_proj)
        V_disease = self.V_disease(disease_proj)
        Q_disease1 = Q_disease.transpose(0, 1)
        K_disease1 = K_disease.transpose(0, 1)
        # drug
        drug_disease = [drug_fea @ (weight(disease_fea)).T for drug_fea, weight, disease_fea in
                        zip(Q_drug1, self.weights_list, K_disease1)]
        drug_disease2 = [drug_fea @ (weight(disease_fea)).T for drug_fea, weight, disease_fea in
                         zip(K_drug1, self.weights_list, Q_disease1)]

        V_drug_disease = [proj1(feat1) for feat1, proj1 in zip(drug_disease, self.drug_disease)]
        V_drug_disease2 = [proj2(feat2) for feat2, proj2 in zip(drug_disease2, self.drug_disease)]
        V_drug_disease_11 = torch.stack(V_drug_disease)
        V_disease_drug_12 = torch.stack(V_drug_disease2)
        V_toall_att_drug = V_drug_disease_11 + V_disease_drug_12  # *

        # disease
        disease_drug = [drug_fea @ (weight(disease_fea)).T for drug_fea, weight, disease_fea in
                        zip(K_disease1, self.weights_list, Q_drug1)]
        disease_drug2 = [drug_fea @ (weight(disease_fea)).T for drug_fea, weight, disease_fea in
                         zip(Q_disease1, self.weights_list, K_drug1)]

        V_disease_drug1 = [proj3(feat3) for feat3, proj3 in zip(disease_drug, self.disease_drug)]
        V_disease_drug2 = [proj4(feat4) for feat4, proj4 in zip(disease_drug2, self.disease_drug)]
        V_disease_drug_9 = torch.stack(V_disease_drug1)
        V_disease_drug_10 = torch.stack(V_disease_drug2)
        V_toall_att_disease = V_disease_drug_9 + V_disease_drug_10  # *

        V_drug_disease_att = V_toall_att_drug.transpose(0, 1)
        V_disease_drug_att = V_toall_att_disease.transpose(0, 1)
        # for drug transformer operation
        V_drug_disease_att = torch.softmax(V_drug_disease_att, dim=-1)
        attn_out_drug = (V_drug_disease_att * V_drug) + V_drug
        # P0-A8 修复：原 .view(num_metapaths, N, hidden) 在行优先布局下会把视图
        # 交错打乱（不是转置），必须用 transpose 才能得到正确的视图切分
        attn_out_drug = attn_out_drug.transpose(0, 1)

        # for disease transformer operation
        V_disease_drug_att = torch.softmax(V_disease_drug_att, dim=-1)
        attn_out_disease = (V_disease_drug_att * V_disease) + V_disease
        attn_out_disease = attn_out_disease.transpose(0, 1)

        drtrans = attn_out_drug          # (M, N, 64)
        ditrans = attn_out_disease       # (M, N, 64)

        homo_graph = homo_graph_cpu.to(device)

        if getattr(self.args, 'use_orig_view', 0):
            # ========== 原版视图方法（创新点二消融基线，--use_orig_view 1） ==========
            # 还原 MAPTrans 原版：硬 Top-2 视图（M=2）→ 每个视图独立过一次 HGT →
            # 与相似度视图拼接 3×gt_out_dim → 纯 Hadamard。
            # （P0 修复保持开启：视图切分用 transpose 而非原版 .view bug）
            dr_hgt_list, di_hgt_list = [], []
            for v in range(self.num_metapaths):
                feature = torch.cat((drtrans[v], ditrans[v], protein_feature_64), dim=0)
                homo_graph.ndata['h'] = feature
                for layer in self.hgt:
                    feature = layer(homo_graph, feature, homo_graph.ndata['_TYPE'],
                                    homo_graph.edata['_TYPE'], presorted=True)
                dr_hgt_list.append(feature[:self.args.drug_number, :])
                di_hgt_list.append(feature[self.args.drug_number:self.args.drug_number + self.args.disease_number, :])
            dr_flat = torch.cat(dr_hgt_list + [dr_sim], dim=1)   # (N_drug, 3×gt_out_dim)
            di_flat = torch.cat(di_hgt_list + [di_sim], dim=1)   # (N_disease, 3×gt_out_dim)
        else:
            # P1-B2：跨视图 Transformer —— 在元路径视图序列上做自注意力（可微视图注意力）
            drtrans_p = drtrans.transpose(0, 1)   # (N, M, 64)
            ditrans_p = ditrans.transpose(0, 1)
            if getattr(self.args, 'use_view_trans', 1):
                drtrans_p = self.drug_trans(drtrans_p.permute(1, 0, 2)).permute(1, 0, 2)
                ditrans_p = self.disease_trans(ditrans_p.permute(1, 0, 2)).permute(1, 0, 2)

            # P1-B1：可微视图注意力池化（替代硬 Top-2；在 HGT 之前合成单一语义视图，
            # 使 HGT 只前向 1 次，避免 4 视图 × HGT 的显存爆炸——24GB 实测 OOM）
            # --use_view_pool 0 时退化为均匀平均池化（无学习权重，创新点二消融对照）
            if getattr(self.args, 'use_view_pool', 1):
                sem_w_d = F.softmax(self.view_attn_sem_drug(drtrans_p), dim=1)   # (N, M, 1)
                sem_w_i = F.softmax(self.view_attn_sem_disease(ditrans_p), dim=1)
                drug_sem = (drtrans_p * sem_w_d).sum(dim=1)       # (N, 64)
                disease_sem = (ditrans_p * sem_w_i).sum(dim=1)    # (N, 64)
            else:
                drug_sem = drtrans_p.mean(dim=1)                  # (N, 64) 均匀平均
                disease_sem = ditrans_p.mean(dim=1)

            # HGT 异构编码（单次前向）
            feature_all = torch.cat((drug_sem, disease_sem, protein_feature_64), dim=0)
            homo_graph.ndata['h'] = feature_all
            for layer in self.hgt:
                hgt_out = layer(homo_graph, feature_all, homo_graph.ndata['_TYPE'],
                                homo_graph.edata['_TYPE'], presorted=True)
                feature_all = hgt_out

            dr_hgt = hgt_out[:self.args.drug_number, :]
            di_hgt = hgt_out[self.args.drug_number:self.args.disease_number + self.args.drug_number, :]

            # 融合 = cat(HGT 输出, 相似度视图) → 2×gt_out_dim
            dr_flat = torch.cat([dr_hgt, dr_sim], dim=1)   # (N_drug, fuse_dim)
            di_flat = torch.cat([di_hgt, di_sim], dim=1)   # (N_disease, fuse_dim)

        # ========== Graph Contrastive Learning ==========
        ssl_loss = torch.tensor(0.0).to(device)
        if self.training and labels is not None:
            # 投影头解耦对比学习空间，避免干扰分类表示
            dr_proj = self.ssl_projector(dr_flat)
            di_proj = self.ssl_projector(di_flat)
            ssl_loss = self.ssl_loss_cross(dr_proj, di_proj, sample, labels)
            # 方向 B：视图间对比（GCGB 风格）——语义视图 vs 相似度视图
            if getattr(self.args, 'use_fuse_ssl', False):
                dr_sim_proj = self.ssl_projector_fuse(dr_sim)
                di_sim_proj = self.ssl_projector_fuse(di_sim)
                ssl_loss = ssl_loss + 0.5 * (
                    self.ssl_loss_views(dr_sim_proj, dr_proj)
                    + self.ssl_loss_views(di_sim_proj, di_proj)
                )

        # P1-B3：配对交互 = Hadamard（+ 可选拼接药物/疾病向量，--use_inter_cat 开关）
        # --use_orig_view 1 时恒为纯 Hadamard（与原版 MAPTrans 一致）
        dr_pair = dr_flat[sample[:, 0]]
        di_pair = di_flat[sample[:, 1]]
        hadamard = torch.mul(dr_pair, di_pair)
        if getattr(self.args, 'use_orig_view', 0) or not getattr(self.args, 'use_inter_cat', 1):
            drdi_embedding = hadamard
        else:
            drdi_embedding = torch.cat([hadamard, dr_pair, di_pair], dim=-1)

        output = self._score_mlp(drdi_embedding)

        return dr_flat, output, ssl_loss
