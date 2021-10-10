'''
@Author: Dong Yi
@Date:2020.9.28
@Description: 这是对CD-LNLP模型的复现
'''

import math
import random
import h5py
import numpy as np
import LNLP_method
import sortscore
import matplotlib.pyplot as plt
from MakeSimilarityMatrix import MakeSimilarityMatrix



if __name__ == '__main__':
    # 定义后面要使用的参数
    alpha = 0.1
    neighbor_rate = 0.9
    weight = 1.0

    # 读取关系数据
    # with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circ2Traits/circRNA_disease.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circad/circrna_disease.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    with h5py.File('./Data/circRNA_disease_noncancer/circRNA_disease_matrix.h5', 'r') as hf:
        circrna_disease_matrix = hf['infor'][:]

    # 划分训练集为五份为后面五折实验做准备
    index_tuple = (np.where(circrna_disease_matrix == 1))
    one_list = list(zip(index_tuple[0], index_tuple[1]))
    random.shuffle(one_list)
    split = math.ceil(len(one_list) / 5)

    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []

    # 5-fold start
    for i in range(0, len(one_list), split):
        test_index = one_list[i:i + split]
        train_index = list(set(one_list)-set(test_index))
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        # 抹除已知关系
        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        rel_matrix = new_circrna_disease_matrix

        prediction_matrix = LNLP_method.linear_neighbor_predict(rel_matrix, alpha, neighbor_rate, weight)
        prediction_matrix = prediction_matrix.A

        aa = prediction_matrix.shape
        bb = roc_circrna_disease_matrix.shape
        zero_matrix = np.zeros((prediction_matrix.shape[0], prediction_matrix.shape[1]))
        print(prediction_matrix.shape)
        print(roc_circrna_disease_matrix.shape)

        score_matrix_temp = prediction_matrix.copy()
        score_matrix = score_matrix_temp + zero_matrix
        minvalue = np.min(score_matrix)
        score_matrix[np.where(roc_circrna_disease_matrix == 2)] = minvalue - 20
        sorted_circrna_disease_matrix, sorted_score_matrix = sortscore.sort_matrix(score_matrix,
                                                                                   roc_circrna_disease_matrix)
        # 下面是正常五折交叉最后整理绘制图像的部分
        tpr_list = []
        fpr_list = []
        recall_list = []
        precision_list = []
        accuracy_list = []
        F1_list = []
        for cutoff in range(sorted_circrna_disease_matrix.shape[0]):
            P_matrix = sorted_circrna_disease_matrix[0:cutoff + 1, :]
            N_matrix = sorted_circrna_disease_matrix[cutoff + 1:sorted_circrna_disease_matrix.shape[0] + 1, :]
            TP = np.sum(P_matrix == 1)
            FP = np.sum(P_matrix == 0)
            TN = np.sum(N_matrix == 0)
            FN = np.sum(N_matrix == 1)
            tpr = TP / (TP + FN)
            fpr = FP / (FP + TN)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            recall_list.append(recall)
            precision_list.append(precision)
            accuracy = (TN + TP) / (TN + TP + FN + FP)
            F1 = (2 * TP) / (2 * TP + FP + FN)
            F1_list.append(F1)
            accuracy_list.append(accuracy)

        # 下面是对top50，top100，top200的预测准确的计数
        top_list = [10, 20, 50, 100, 200]
        for num in top_list:
            P_matrix = sorted_circrna_disease_matrix[0:num, :]
            N_matrix = sorted_circrna_disease_matrix[num:sorted_circrna_disease_matrix.shape[0] + 1, :]
            top_count = np.sum(P_matrix == 1)
            print("top" + str(num) + ": " + str(top_count))

        all_tpr.append(tpr_list)
        all_fpr.append(fpr_list)
        all_recall.append(recall_list)
        all_precision.append(precision_list)
        all_accuracy.append(accuracy_list)
        all_F1.append(F1_list)

    tpr_arr = np.array(all_tpr)
    fpr_arr = np.array(all_fpr)
    recall_arr = np.array(all_recall)
    precision_arr = np.array(all_precision)
    accuracy_arr = np.array(all_accuracy)
    F1_arr = np.array(all_F1)

    mean_cross_tpr = np.mean(tpr_arr, axis=0)  # axis=0
    mean_cross_fpr = np.mean(fpr_arr, axis=0)
    mean_cross_recall = np.mean(recall_arr, axis=0)
    mean_cross_precision = np.mean(precision_arr, axis=0)
    mean_cross_accuracy = np.mean(accuracy_arr, axis=0)
    # 计算此次五折的平均评价指标数值
    mean_accuracy = np.mean(np.mean(accuracy_arr, axis=1), axis=0)
    mean_recall = np.mean(np.mean(recall_arr, axis=1), axis=0)
    mean_precision = np.mean(np.mean(precision_arr, axis=1), axis=0)
    mean_F1 = np.mean(np.mean(F1_arr, axis=1), axis=0)
    print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f" % (mean_accuracy, mean_recall, mean_precision, mean_F1))

    roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)
    AUPR = np.trapz(mean_cross_precision, mean_cross_recall)
    print("AUC:%.4f,AUPR:%.4f" % (roc_auc, AUPR))

    # 存储tpr，fpr,recall,precision
    with h5py.File('./PlotFigure/CD-LNLP_noncancer_5fold_AUC.h5','w') as hf:
        hf['fpr'] = mean_cross_fpr
        hf['tpr'] = mean_cross_tpr
    with h5py.File('./PlotFigure/CD-LNLP_noncacner_5fold_AUPR.h5','w') as h:
        h['recall'] = mean_cross_recall
        h['precision'] = mean_cross_precision

    plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean AUC=%0.4f' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)
    # plt.savefig("./FinalResultPng/roc-circad_10fold.png")
    print("runtime over, now is :")
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    plt.show()

#=========================================下面是衡量matrix所有预测值top预测准确个数的代码============================================#
# # 这里把这个得分矩阵进行整理，找到top-20,top-50,top-100，预测准确的数量，但是要除开训练集
# # 先将二维数组变为一维
# prediction_array = prediction_matrix.flatten()
# sort_index = np.argsort(-prediction_array)
# # 对这个index进行处理推出属于prediction的哪一行哪一列
# top_tuple = []
# for i in range(len(sort_index)):
#     row = sort_index[i] // prediction_matrix.shape[1]
#     col = sort_index[i] % prediction_matrix.shape[1]
#     if (row, col) not in train_index:
#         top_tuple.append((row, col))
# top_20 = 0
# top_50 = 0
# top_100 = 0
# top_150 = 0
# top_200 = 0
#
# top_20_list = top_tuple[:20]
# top_50_list = top_tuple[:50]
# top_100_list = top_tuple[:100]
# top_150_list = top_tuple[:150]
# top_200_list = top_tuple[:200]
#
# for (u, i) in top_20_list:
#     if circrna_disease_matrix[u, i] == 1:
#         top_20 += 1
#
# for (u, i) in top_50_list:
#     if circrna_disease_matrix[u, i] == 1:
#         top_50 += 1
#
# for (u, i) in top_100_list:
#     if circrna_disease_matrix[u, i] == 1:
#         top_100 += 1
#
# for (u, i) in top_150_list:
#     if circrna_disease_matrix[u, i] == 1:
#         top_150 += 1
#
# for (u, i) in top_200_list:
#     if circrna_disease_matrix[u, i] == 1:
#         top_200 += 1
#
# print("top_20:" + str(top_20))
# print("top_50:" + str(top_50))
# print("top_100:" + str(top_100))
# print("top_150:" + str(top_150))
# print("top_200:" + str(top_200))





