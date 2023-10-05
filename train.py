"""
@author: Yyujie
@email: yyj17320071233@163.com
@Date: 2023/3/26 13:28
@Description
"""
import numpy as np
import random
import torch
import pandas as pd
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, auc
from model import LayerGCN



class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, CDA_ass_reconstruct, CD_asso_train, pos_x_index, pos_y_index, neg_x_index,
                neg_y_index, alpha):
        loss_fn = torch.nn.MSELoss(reduction='none')  # 均方损失 # ‘none’ 'mean' 'sum'  reduction：维度要不要缩减或如何缩减
        CD_asso_train = torch.from_numpy(CD_asso_train)
        loss_mat = loss_fn(CDA_ass_reconstruct.float(), CD_asso_train.float())  # CDA_ass_reconstruct -->预测分数矩阵
        # 仅计算训练损失
        loss = (loss_mat[pos_x_index, pos_y_index].sum() * (1 - alpha) + loss_mat[
            neg_x_index, neg_y_index].sum() * alpha) / (2 * 2149)
        return loss


def evaluate(model, A_stack_test, test, lnc_sim, dis_sim, miR_asso):
    model.eval()  # 切换评估模式
    pred_list = []
    ground_truth = []
    # with：对资源进行访问，使用后释放资源
    with torch.no_grad():
        # 在该模块下，所有计算得出的tensor的requires_grad自动设置为False，即反向传播时不会自动求导
        prediction_score = model(A_stack_test, lnc_sim, dis_sim, miR_asso)

    prediction_score = prediction_score.numpy()

    for ele in test:
        '''
        测试集的预测列表和真实标签列表
        '''
        pred_list.append(prediction_score[ele[0], ele[1]])
        ground_truth.append(ele[2])

    fpr, tpr, _ = roc_curve(ground_truth, pred_list)
    AUROC = auc(fpr, tpr)
    precision1, recall1, _ = precision_recall_curve(ground_truth, pred_list)
    aupr = auc(recall1, precision1)

    result = [0 if j < 0.5 else 1 for j in pred_list]
    accuracy = accuracy_score(ground_truth, result)
    precision = precision_score(ground_truth, result)
    recall = recall_score(ground_truth, result)
    f1 = f1_score(ground_truth, result)

    return AUROC, aupr, accuracy, precision, recall, precision1, recall1, f1, fpr, tpr


def train_evaluate(LD_asso_train, test, A_stack_train, A_stack_test, lnc_sim, dis_sim, miR_sim, pos_x_index,
                   pos_y_index, neg_x_index, neg_y_index, circRNA_num, disease_num, miRNA_num, args):
    # 设置随机种子参数
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    pos_x_index = torch.tensor(pos_x_index, dtype=torch.long)
    pos_y_index = torch.tensor(pos_y_index, dtype=torch.long)
    neg_x_index = torch.tensor(neg_x_index, dtype=torch.long)  # 从list创建一个张量
    neg_y_index = torch.tensor(neg_y_index, dtype=torch.long)
    A_stack_train = torch.from_numpy(A_stack_train).long()
    A_stack_test = torch.from_numpy(A_stack_test).long()  # 从numpy.ndarray创建一个张量

    # 获得参数
    lr = args.lr
    weight_decay = args.weight_decay
    alpha = args.alpha
    latent_dim = args.latent_dim
    layer_num = args.layer_num
    dropout = args.dropout
    epochs = args.epochs

    loss_fun = Loss()
    # 调用模型
    model = LayerGCN(circRNA_num, disease_num, miRNA_num, latent_dim, layer_num, dropout)

    # ablation experiment
    # model = GCN(circRNA_num, disease_num, miRNA_num, latent_dim, layer_num, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        predict_score = model(A_stack_train, lnc_sim, dis_sim, miR_sim)
        # LD_asso_train = torch.from_numpy(LD_asso_train)
        loss = loss_fun(predict_score, LD_asso_train, pos_x_index, pos_y_index, neg_x_index, neg_y_index, alpha)
        optimizer.zero_grad()  # 清空之前保留的梯度信息
        loss.backward()  # 将mini_batch的信息反传回去
        optimizer.step()  # 根据optimizer参数和梯度更新参数 w.data-=w.grad*lr

        val_auc, val_aupr, accuracy, precision, recall, precision1, recall1, f1, fpr, tpr = evaluate(model,
                                                                                                     A_stack_test, test,
                                                                                                     lnc_sim, dis_sim,
                                                                                                     miR_sim)
        print('Epoch {:d} | Train Loss {:.4f} |'
              'val_auc {:.4f} | val_aupr {:.4f} |'
              'val_precision {:.4f} | val_recall {:.4f} |'
              'val_accuracy {:.4f} | val_f1 {:.4f}'
              .format(epoch + 1, loss.item(), val_auc, val_aupr, precision, recall, accuracy, f1))
    test_auc, test_aupr, acc_test, pre_test, recall_test, pre1_test, recall1_test, f1_test, fpr, tpr = evaluate(model,
                                                                                                                A_stack_test,
                                                                                                                test,
                                                                                                                lnc_sim,
                                                                                                                dis_sim,
                                                                                                                miR_sim)
    prediction_score = predict_score.detach().numpy()
    prediction_score_pd = pd.DataFrame(prediction_score)
    prediction_score_pd.to_csv('./savedata/Predscore.csv', index=False, header=None)

    return test_auc, test_aupr, pre_test, recall_test, pre1_test, recall1_test, acc_test, f1_test, fpr, tpr
