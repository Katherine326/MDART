import os

import numpy as np
import random
import torch
import pandas as pd
import dgl
import networkx as nx
from sklearn.model_selection import StratifiedKFold
import scipy.sparse as sp
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()
    values = torch.ones(len(edges))
    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    return adj


def k_matrix(matrix, k):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)


def gdc_diffusion(A, alpha=0.05, order=3, k=20):
    """
    GDC（Graph Diffusion Convolution）扩散核。

    将单尺度 KNN 相似度图扩展为多尺度扩散图，捕获 drug-drug / disease-disease
    关系网络中的长程依赖（对应 HGCL-DR / MGTAL-DR 的图扩散机制）。

    S = Σ_{i=0}^{order} alpha * (1-alpha)^i * A_norm^i
    其中 A_norm 为对称归一化邻接矩阵；随后做 top-k 行稀疏化、对称化并补自环。

    Args:
        A:      KNN 相似度矩阵（含自环）
        alpha:  跃迁概率（控制扩散强度）
        order:  扩散阶数（多尺度范围）
        k:      稀疏化后每行保留的邻居数

    Returns:
        扩散后的稀疏相似度矩阵
    """
    n = A.shape[0]
    A = np.asarray(A, dtype=np.float64)
    np.fill_diagonal(A, 0.0)  # 扩散输入去掉自环（扩散自然包含）

    d = A.sum(axis=1) + 1e-6
    d_inv_sqrt = 1.0 / np.sqrt(d)
    A_norm = (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]

    S = np.zeros_like(A_norm)
    A_pow = np.eye(n)
    for i in range(order + 1):
        if i > 0:
            A_pow = A_pow @ A_norm
        S += alpha * (1.0 - alpha) ** i * A_pow

    # top-k 稀疏化 + 对称化 + 自环
    S = np.maximum(S, S.T)
    np.fill_diagonal(S, 0.0)
    idx = np.argsort(-S, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    S_sparse = np.zeros_like(S)
    S_sparse[rows, idx] = S[rows, idx]
    S_sparse = np.maximum(S_sparse, S_sparse.T)
    np.fill_diagonal(S_sparse, 1.0)
    return S_sparse


def get_data(args):
    data = dict()

    drf = pd.read_csv(args.data_dir + 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()
    drg = pd.read_csv(args.data_dir + 'DrugGIP.csv').iloc[:, 1:].to_numpy()

    dip = pd.read_csv(args.data_dir + 'DiseasePS.csv').iloc[:, 1:].to_numpy()
    dig = pd.read_csv(args.data_dir + 'DiseaseGIP.csv').iloc[:, 1:].to_numpy()

    data['drug_number'] = int(drf.shape[0])
    data['disease_number'] = int(dig.shape[0])

    data['drf'] = drf
    data['drg'] = drg
    data['dip'] = dip
    data['dig'] = dig

    data['drdi'] = pd.read_csv(args.data_dir + 'DrugDiseaseAssociationNumber.csv', dtype=int).to_numpy()
    data['drpr'] = pd.read_csv(args.data_dir + 'DrugProteinAssociationNumber.csv', dtype=int).to_numpy()
    data['dipr'] = pd.read_csv(args.data_dir + 'DiseaseProteinAssociationNumber.csv', dtype=int).to_numpy()
    data['didr'] = pd.read_csv(args.data_dir + 'DiseaseDrugAssociationNumber.csv', dtype=int).to_numpy()
    data['prdr'] = pd.read_csv(args.data_dir + 'ProteinDrugAssociationNumber.csv', dtype=int).to_numpy()
    data['prdi'] = pd.read_csv(args.data_dir + 'ProteinDiseaseAssociationNumber.csv', dtype=int).to_numpy()
    # data['drugfeature'] = pd.read_csv('./data/B-dataset/drug_drug_baseline.csv', header=None).values
    # data['diseasefeature'] = pd.read_csv('./data/B-dataset/disease_disease_baseline.csv', header=None).values
    data['drugfeature'] = pd.read_csv(args.data_dir + 'Drug_mol2vec.csv', header=None).iloc[:, 1:].to_numpy()
    data['diseasefeature'] = pd.read_csv(args.data_dir + 'DiseaseFeature.csv', header=None).iloc[:, 1:].to_numpy()
    data['proteinfeature'] = pd.read_csv(args.data_dir + 'Protein_ESM.csv', header=None).iloc[:, 1:].to_numpy()
    data['protein_number']= data['proteinfeature'].shape[0]
    num_drugs = data['drug_number']
    num_diseases = data['disease_number']
    num_proteins = data['protein_number']

    adj_matrices = {}

    # 1. 药物-疾病关联 (drug-disease)
    adj_matrices['drug-disease'] = np.zeros((num_drugs, num_diseases), dtype=int)
    for i, j in data['drdi']:
        adj_matrices['drug-disease'][i, j] = 1  # 药物到疾病的关联

    # 2. 药物-蛋白质关联 (drug-protein)
    adj_matrices['drug-protein'] = np.zeros((num_drugs, num_proteins), dtype=int)
    for i, j in data['drpr']:
        adj_matrices['drug-protein'][i, j] = 1  # 药物到蛋白质的关联

    # 3. 疾病-蛋白质关联 (disease-protein)
    adj_matrices['disease-protein'] = np.zeros((num_diseases, num_proteins), dtype=int)
    for i, j in data['dipr']:
        adj_matrices['disease-protein'][i, j] = 1  # 疾病到蛋白质的关联

    # 4. 疾病-药物关联 (disease-drug)
    adj_matrices['disease-drug'] = np.zeros((num_diseases, num_drugs), dtype=int)
    for i, j in data['didr']:
        adj_matrices['disease-drug'][i, j] = 1  # 疾病到药物的关联

    # 5. 蛋白质-药物关联 (protein-drug)
    adj_matrices['protein-drug'] = np.zeros((num_proteins, num_drugs), dtype=int)
    for i, j in data['prdr']:
        adj_matrices['protein-drug'][i, j] = 1  # 蛋白质到药物的关联

    # 6. 蛋白质-疾病关联 (protein-disease)
    adj_matrices['protein-disease'] = np.zeros((num_proteins, num_diseases), dtype=int)
    for i, j in data['prdi']:
        adj_matrices['protein-disease'][i, j] = 1  # 蛋白质到疾病的关联



    drug_disease_adj = adj_matrices['drug-disease']

    return data, adj_matrices

def data_processing(data, args):
    drdi_matrix = get_adj(data['drdi'], (args.drug_number, args.disease_number))
    one_index = []
    zero_index = []
    for i in range(drdi_matrix.shape[0]):
        for j in range(drdi_matrix.shape[1]):
            if drdi_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)

    # 将所有负样本保存在 unsample 中备用
    data['unsample'] = np.array(zero_index)

    # 使用全部负样本
    index = np.array(one_index + zero_index, dtype=int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    label_p = np.array([1] * len(one_index), dtype=int)

    drdi_p = samples[samples[:, 2] == 1, :]
    drdi_n = samples[samples[:, 2] == 0, :]

    drs_mean = (data['drf'] + data['drg']) / 2
    dis_mean = (data['dip'] + data['dig']) / 2

    drs = np.where(data['drf'] == 0, data['drg'], drs_mean)
    dis = np.where(data['dip'] == 0, data['dip'], dis_mean)

    data['drs'] = drs
    data['dis'] = dis
    data['all_samples'] = samples
    data['all_drdi'] = samples[:, :2]
    data['all_drdi_p'] = drdi_p
    data['all_drdi_n'] = drdi_n
    data['all_label'] = label
    data['all_label_p'] = label_p
    return data






def k_fold(data, args):
    k = args.k_fold
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    X = data['all_drdi']
    Y = data['all_label']
    # n = skf.get_n_splits(X, Y)
    X_train_all, X_test_all, Y_train_all, Y_test_all = [], [], [], []
    for train_index, test_index in skf.split(X, Y):
        # print('Train:', train_index, 'Test:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_train = np.expand_dims(Y_train, axis=1).astype('float64')
        Y_test = np.expand_dims(Y_test, axis=1).astype('float64')
        X_train_all.append(X_train)
        X_test_all.append(X_test)
        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)

    for i in range(k):
        dir_path = os.path.join(args.data_dir, 'fold', str(i))


        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        X_train1 = pd.DataFrame(data=np.concatenate((X_train_all[i], Y_train_all[i]), axis=1), columns=['drug', 'disease', 'label'])
        X_train1.to_csv(args.data_dir + 'fold/' + str(i) + '/data_train.csv')
        X_test1 = pd.DataFrame(data=np.concatenate((X_test_all[i], Y_test_all[i]), axis=1), columns=['drug', 'disease', 'label'])
        X_test1.to_csv(args.data_dir + 'fold/' + str(i) + '/data_test.csv')

    data['X_train'] = X_train_all
    data['X_test'] = X_test_all
    data['Y_train'] = Y_train_all
    data['Y_test'] = Y_test_all
    return data



def dgl_similarity_graph(data, args):
    drdr_matrix = k_matrix(data['drs'], args.neighbor)
    didi_matrix = k_matrix(data['dis'], args.neighbor)
    # 方向 A：GDC 图扩散，将单尺度 KNN 图扩展为多尺度扩散图（捕获长程依赖）
    if getattr(args, 'use_gdc', False):
        drdr_matrix = gdc_diffusion(drdr_matrix, getattr(args, 'gdc_alpha', 0.05),
                                    getattr(args, 'gdc_order', 3), args.neighbor)
        didi_matrix = gdc_diffusion(didi_matrix, getattr(args, 'gdc_alpha', 0.05),
                                    getattr(args, 'gdc_order', 3), args.neighbor)
    drdr_nx = nx.from_numpy_array(drdr_matrix)
    didi_nx = nx.from_numpy_array(didi_matrix)
    drdr_graph = dgl.from_networkx(drdr_nx)
    didi_graph = dgl.from_networkx(didi_nx)

    # P1-B5：相似度图节点特征由融合均值升级为多视图拼接（drf|drg, dip|dig）。
    # 图拓扑仍由原 KNN/GDC 结构决定（k_matrix 作用于融合后的 data['drs']），
    # 仅节点特征变宽；--use_multi_sim_feat=0 时退回旧行为。
    if getattr(args, 'use_multi_sim_feat', 1):
        drdr_graph.ndata['drs'] = torch.tensor(
            np.concatenate([data['drf'], data['drg']], axis=1).astype(np.float32))
        didi_graph.ndata['dis'] = torch.tensor(
            np.concatenate([data['dip'], data['dig']], axis=1).astype(np.float32))
    else:
        drdr_graph.ndata['drs'] = torch.tensor(data['drs'])
        didi_graph.ndata['dis'] = torch.tensor(data['dis'])

    return drdr_graph, didi_graph, data

def loocv_split_cold_start(data, index, device):
    """
    LOOCV 冷启动划分。

    参数:
        data: 包含 'all_drdi' (样本) 和 'all_label' (标签) 的字典
        index: 测试样本索引 (指定冷启动药物或疾病)
        device: 目标设备 (如 'cuda' 或 'cpu')

    返回:
        训练数据 (X_train, Y_train) 和测试数据 (X_test, Y_test)
    """
    # 获取所有数据
    X = torch.tensor(data['all_drdi'])  # 样本对 (drug, disease)
    Y = torch.tensor(data['all_label'])  # 对应标签



    # 当前测试样本的药物和疾病
    test_drug, test_disease = X[index]

    # 测试集：包含所有与目标药物或疾病相关的样本
    test_indices = [
        i for i in range(len(X))
        if X[i][0] == test_drug or X[i][1] == test_disease
    ]
    X_test = X[test_indices]
    Y_test = Y[test_indices]

    # 训练集：移除所有测试样本
    train_indices = [
        i for i in range(len(X))
        if i not in test_indices
    ]
    X_train = X[train_indices]
    Y_train = Y[train_indices]

    # 转换为张量并返回
    X_train = torch.LongTensor(X_train)
    Y_train = torch.LongTensor(np.expand_dims(Y_train, axis=1))
    X_test = torch.LongTensor(X_test).to(device)
    Y_test = torch.LongTensor(np.expand_dims(Y_test, axis=1)).to(device)

    return X_train, Y_train, X_test, Y_test

def dgl_heterograph(data, drdi, args):
    drdi_list, drpr_list, dipr_list, prdi_list, didr_list, prdr_list = [], [], [], [], [], []


    for i in range(drdi.shape[0]):
        drdi_list.append(drdi[i])
    for i in range(data['drpr'].shape[0]):
        drpr_list.append(data['drpr'][i])
    for i in range(data['dipr'].shape[0]):
        dipr_list.append(data['dipr'][i])
    for i in range(data['prdi'].shape[0]):
        prdi_list.append(data['prdi'][i])
    for i in range(data['didr'].shape[0]):
        didr_list.append(data['didr'][i])
    for i in range(data['prdr'].shape[0]):
        prdr_list.append(data['prdr'][i])


    node_dict = {
        'drug': args.drug_number,
        'disease': args.disease_number,
        'protein': args.protein_number
    }


    heterograph_dict = {
        ('drug', 'association', 'disease'): (drdi_list),
        ('drug', 'association', 'protein'): (drpr_list),
        ('disease', 'association', 'protein'): (dipr_list),
    }


    data['feature_dict'] = {
        'drug': torch.tensor(data['drugfeature']),
        'disease': torch.tensor(data['diseasefeature']),
        'protein': torch.tensor(data['proteinfeature'])
    }
    meta_paths_drug = {
        "drug-disease-drug": [
            ('drug', 'drug-disease', 'disease'),
            ('disease', 'disease-drug', 'drug')
        ],
        "drug-protein-drug": [
            ('drug', 'drug-protein', 'protein'),
            ('protein', 'protein-drug', 'drug')
        ],
        "drug-disease-protein-drug": [
            ('drug', 'drug-disease', 'disease'),
            ('disease', 'disease-protein', 'protein'),
            ('protein', 'protein-drug', 'drug')
        ],
        "drug-protein-disease-drug": [
            ('drug', 'drug-protein', 'protein'),
            ('protein', 'protein-disease', 'disease'),
            ('disease', 'disease-drug', 'drug')
        ],
        # "drug-disease-protein-disease-drug": [
        #     ('drug', 'drug-disease', 'disease'),
        #     ('disease', 'disease-protein', 'protein'),
        #     ('protein', 'protein-disease', 'disease'),
        #     ('disease', 'disease-drug', 'drug')
        # ],
    }
    meta_paths_disease = {
        "disease-protein-disease": [
            ('disease', 'disease-protein', 'protein'),
            ('protein', 'protein-disease', 'disease')
        ],
        "disease-drug-disease": [
            ('disease', 'disease-drug', 'drug'),
            ('drug', 'drug-disease', 'disease')
        ],
        "disease-protein-drug-disease": [
            ('disease', 'disease-protein', 'protein'),
            ('protein', 'protein-drug', 'drug'),
            ('drug', 'drug-disease', 'disease')
        ],
        "disease-drug-protein-disease": [
            ('disease', 'disease-drug', 'drug'),
            ('drug', 'drug-protein', 'protein'),
            ('protein', 'protein-disease', 'disease')
        ],
        # "disease-drug-protein-drug-disease": [
        #     ('disease', 'disease-drug', 'drug'),
        #     ('drug', 'drug-protein', 'protein'),
        #     ('protein', 'protein-drug', 'drug'),
        #     ('drug', 'drug-disease', 'disease')
        # ],
    }
    # 创建异构图
    drdipr_graph = dgl.heterograph(heterograph_dict, num_nodes_dict=node_dict)

    return drdipr_graph, data, meta_paths_drug, meta_paths_disease






