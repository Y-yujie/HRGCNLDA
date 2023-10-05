import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from uitls import load_data, get_train_test, relations_to_matrix
from train import train_evaluate
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from numpy import interp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--kfolds', default=5, type=int)  # 5>10
    parser.add_argument('--ratio', default='one', type=str)  # ten>all>one

    parser.add_argument('--layer_num', default=4, type=int)  # 最好的结果
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--latent_dim', default=128, type=int)

    parser.add_argument('--lr', default=0.01, type=float)  # 最好的结果
    parser.add_argument('--alpha', default=0.6, type=float)
    parser.add_argument('--weight_decay', default=1e-7, type=float)
    parser.add_argument('--random_seed', default=42, type=int)
    args = parser.parse_args()
    print(args)
    lncRNA_miRNA_intera, miRNA_disease_asso, dataset, lnc_sim, dis_sim, miR_sim = load_data(r'D:\MyData\Data', args)
    lncRNA_num = lnc_sim.shape[0]
    disease_num = dis_sim.shape[0]
    miRNA_num = miR_sim.shape[0]

    lnc_sim = torch.FloatTensor(lnc_sim)
    dis_sim = torch.FloatTensor(dis_sim)
    miR_sim = torch.FloatTensor(miR_sim)

    auc_result = []
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []
    aupr_result = []

    fprs = []
    tprs = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 240)

    precisions = []
    mean_recall = np.linspace(0, 1, 240)

    # 用于绘制每折训练的ROC曲线
    kf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=42)
    fold = 1

    # 交叉验证
    for train_index, test_index in kf.split(dataset[:, :2], dataset[:, 2]):
        train, test = dataset[train_index], dataset[test_index]  # train: data_set 4/5
        print("#############%d fold" % fold + "#############")

        LD_asso_train, LD_asso_test, pos_x_index, pos_y_index, neg_x_index, neg_y_index = get_train_test(train, test,
                                                                                                         lncRNA_num,
                                                                                                         disease_num)
        A_stack_train = relations_to_matrix(LD_asso_train, lncRNA_miRNA_intera, miRNA_disease_asso, lnc_sim, dis_sim,
                                            miR_sim)
        A_stack_test = relations_to_matrix(LD_asso_test, lncRNA_miRNA_intera, miRNA_disease_asso, lnc_sim, dis_sim,
                                           miR_sim)
        test_auc, test_aupr, pre_test, recall_test, pre1_test, recall1_test, acc_test, f1_test, fpr, tpr = train_evaluate(
            LD_asso_train, test, A_stack_test, A_stack_train, lnc_sim, dis_sim, miR_sim, pos_x_index, pos_y_index,
            neg_x_index, neg_y_index, lncRNA_num, disease_num, miRNA_num, args)

        roc_auc = auc(fpr, tpr)
        plt.figure(1)
        plt.plot(fpr, tpr, label='ROC fold %d (AUC = %0.4f)' % (fold, roc_auc))
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

        roc_aupr = auc(recall1_test, pre1_test)
        plt.figure(2)
        plt.plot(recall1_test, pre1_test, label='ROC fold %d (AUPR = %0.4f)' % (fold, roc_aupr))
        precisions.append(interp(mean_recall, np.array(list(recall1_test[::-1])), np.array(list(pre1_test[::-1]))))

        auc_result.append(test_auc)
        aupr_result.append(test_aupr)
        acc_result.append(acc_test)
        pre_result.append(pre_test)
        recall_result.append(recall_test)
        f1_result.append(f1_test)
        # fprs.append(fpr)
        # tprs.append(tpr)
        fold = fold + 1

    mean_tpr /= 5
    mean_tpr[-1] = 1.0

    mean_precision = np.mean(precisions, axis=0)
    mean_AUC = auc(mean_fpr, mean_tpr)
    mean_AUPR = auc(mean_recall, mean_precision)

    plt.figure(1)
    plt.plot(mean_fpr, mean_tpr, label='mean ROC=%0.4f' % mean_AUC)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc=0)
    # plt.savefig('./savedata/LRGCNLDA_ROC.tif', dpi=300)

    # save ablation experiment figure
    # plt.savefig('./savedata/GCNLDA_ROC.tif', dpi=300)

    plt.figure(2)
    plt.plot(mean_recall, mean_precision, label='AUPR ROC = %0.4f' % mean_AUPR)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc=0)
    # plt.savefig('./savedata/LRGCNLDA_AUPR.tif', dpi=300)

    # save ablation experiment figure
    # plt.savefig('./savedata/GCNLDA_AUPR.tif', dpi=300)

    plt.show()

    print('-AUC mean: %.4f, variance: %.4f \n' % (mean_AUC, np.std(auc_result)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
          'aupr mean: %.4f, variance: %.4f \n' % (mean_AUPR, np.std(aupr_result)),
          )
