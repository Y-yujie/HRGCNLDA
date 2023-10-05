import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp


def load_data(data_path, args):
    # Dataset4
    lncRNA_disease_association = pd.read_excel("./dataset/Dataset4_lncRNA_disease_association_240x405.xlsx",
                                               header=None)
    lncRNA_miRNA_interaction = pd.read_excel("./dataset/Dataset4_lncRNA_miRNA interaction_240x495.xlsx", header=None)
    miRNA_disease_association = pd.read_excel("./dataset/Dataset4_miRNA_disease association_495x405.xlsx", header=None)
    lncRNA_func_sim = pd.read_excel(data_path + r"\Dataset5\04-lncRNA-lncRNA.xlsx", header=None)
    lncRNA_Gkl_sim = pd.read_excel("./dataset/Dataset4_lncRNA_Gkl_240.xlsx", header=None)
    miRNA_seq_sim = pd.read_csv(data_path + r"\Dataset5\09-miRNA_sequence_similarity.csv", header=None)
    miRNA_Gkl_sim = pd.read_excel("./dataset/Dataset4_miRNA_Gkl_495.xlsx", header=None)
    dis_sem_sim = pd.read_excel("./dataset/Dataset4_disease_semantic similarity_405.xlsx", header=None)
    dis_Gkl_sim_L = pd.read_excel("./dataset/Dataset4_disease_Gkl_L405.xlsx", header=None)
    dis_Gkl_sim_M = pd.read_excel("./dataset/Dataset4_disease_Gkl_M405.xlsx", header=None)


    # lncRNA相似性融合
    lncRNA_func_sim = np.array(lncRNA_func_sim)
    lncRNA_Gkl_sim = np.array(lncRNA_Gkl_sim)
    lncRNA_sim = np.zeros((lncRNA_func_sim.shape[0], lncRNA_func_sim.shape[1]))
    for i in range(lncRNA_func_sim.shape[0]):
        for j in range(lncRNA_func_sim.shape[1]):
            if lncRNA_func_sim[i][j] == 0:
                lncRNA_sim[i][j] = lncRNA_Gkl_sim[i][j]
            else:
                lncRNA_sim[i][j] = lncRNA_func_sim[i][j]

    # miRNA相似性融合
    miRNA_seq_sim = np.array(miRNA_seq_sim)
    miRNA_Gkl_sim = np.array(miRNA_Gkl_sim)
    miRNA_sim = np.zeros(miRNA_seq_sim.shape)
    for i in range(miRNA_seq_sim.shape[0]):
        for j in range(miRNA_seq_sim.shape[1]):
            if miRNA_seq_sim[i][j] == 0:
                miRNA_sim[i][j] = miRNA_Gkl_sim[i][j]
            else:
                miRNA_sim[i][j] = miRNA_seq_sim[i][j]

    # disease相似性融合
    dis_semantic_sim = np.array(dis_sem_sim)
    dis_Gkl_sim_L = np.array(dis_Gkl_sim_L)
    dis_Gkl_sim_M = np.array(dis_Gkl_sim_M)
    dis_sim = np.zeros(shape=(dis_semantic_sim.shape[0], dis_semantic_sim.shape[1]))
    for i in range(dis_semantic_sim.shape[0]):
        for j in range(dis_semantic_sim.shape[1]):
            if dis_semantic_sim[i][j] == 0:
                dis_sim[i][j] = (dis_Gkl_sim_L[i][j] + dis_Gkl_sim_M[i][j]) / 2
            else:
                dis_sim[i][j] = dis_semantic_sim[i][j]

    whole_positive_index = []  # 正样本指针(i,j)
    whole_negative_index = []

    lncRNA_disease_asso = np.array(lncRNA_disease_association)
    miRNA_disease_asso = np.array(miRNA_disease_association)
    lncRNA_miRNA_intera = np.array(lncRNA_miRNA_interaction)
    lncRNA_sim = np.array(lncRNA_sim)
    miRNA_sim = np.array(miRNA_sim)
    dis_sim = np.array(dis_sim)

    for i in range(len(lncRNA_disease_asso)):
        for j in range(len(lncRNA_disease_asso[0])):
            if lncRNA_disease_asso[i][j] == 1:
                whole_positive_index.append([i, j])
            else:
                whole_negative_index.append([i, j])

    negative_sample_index = []
    # 对负样本采样，ratio为负正样本比例,因为负样本比较多
    if args.ratio == 'ten':
        '''
                np.random.choice(A,size,replace):从给定的一维数组中随机采样
                A:给定的一维数组
                size:采样结果的数量
                replace:采样的样本是否要更换，指定为True采样元素有重复，False则采样不会有重复
                '''
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 # np.arange(),返回一个有终点和起点的固定步长的排列，用在此处和range()效果相同
                                                 size=10 * len(whole_positive_index), replace=False)
    elif args.ratio == 'one':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 size=1 * len(whole_positive_index), replace=False)
    elif args.ratio == 'all':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)), size=len(whole_negative_index),
                                                 replace=False)
    else:
        print('wrong positive negative ratio')

    dataset = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    count = 0
    for i in whole_positive_index:
        dataset[count][0] = i[0]
        dataset[count][1] = i[1]
        dataset[count][2] = 1
        count += 1
    for i in negative_sample_index:
        dataset[count][0] = whole_negative_index[i][0]
        dataset[count][1] = whole_negative_index[i][1]
        dataset[count][2] = 0
        count += 1

    return lncRNA_miRNA_intera, miRNA_disease_asso, dataset, lncRNA_sim, dis_sim, miRNA_sim


def get_train_test(train, test, lncRNA_num, disease_num):
    asso_train = np.zeros((lncRNA_num, disease_num))
    asso_test = np.zeros((lncRNA_num, disease_num))
    pos_x_index = []
    pos_y_index = []
    neg_x_index = []
    neg_y_index = []
    for ele in train:
        asso_train[ele[0], ele[1]] = ele[2]  # 只有用作训练的数据中的正样本为1，其余值均为0
        if ele[2] == 1:
            pos_x_index.append(ele[0])
            pos_y_index.append(ele[1])  # 存储的是用作训练的数据中正样本的索引
        elif ele[2] == 0:
            neg_x_index.append(ele[0])
            neg_y_index.append(ele[1])  # 存储的是用作训练的数据中负样本的索引

    for ele in test:
        asso_test[ele[0], ele[1]] = ele[2]

    return asso_train, asso_test, pos_x_index, pos_y_index, neg_x_index, neg_y_index


def re_normalization(A):
    A = A + torch.eye(A.size()[0])  # 加上自环
    D_vector = torch.sum(A, dim=1)
    D = np.power(D_vector, -0.5) * torch.eye(A.size()[0])
    A_p = torch.mm(torch.mm(D, A), D)
    return A_p


def relations_to_matrix(lncRNA_disease_asso, lncRNA_miRNA_intera, miRNA_disease_asso, lnc_sim, dis_sim, miR_sim):
    # 返回一个堆叠的邻接矩阵
    h1 = np.hstack((np.hstack((lnc_sim, lncRNA_disease_asso)), lncRNA_miRNA_intera))
    h2 = np.hstack((np.hstack((lncRNA_disease_asso.T, dis_sim)), miRNA_disease_asso.T))
    h3 = np.hstack((np.hstack((lncRNA_miRNA_intera.T, miRNA_disease_asso)), miR_sim))
    A = np.vstack((np.vstack((h1, h2)), h3))

    return A
