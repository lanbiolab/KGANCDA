'''
@Date：2020.8.3
@Author：DongYi
@Description：针对DWNN-RLS方法进行逐步的实现
'''
import math
import random
import h5py as h5py
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import sortscore
from MakeSimilarityMatrix import MakeSimilarityMatrix

if __name__ == '__main__':
    # 读取关系数据
    # with h5py.File('./Data/disease-circRNA.h5','r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_miRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circ2Traits/circRNA_disease.h5','r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circad/circrna_disease.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # noncancer_disease
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
    for i in range(0,len(one_list),split):
        test_index = one_list[i:i+split]
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        # 抹除已知关系
        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        rel_matrix = new_circrna_disease_matrix
        # 计算相似高斯相似矩阵
        make_sim_matrix = MakeSimilarityMatrix(rel_matrix)
        circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix,make_sim_matrix.dissimmatrix
        # # 读取疾病的语义相似性
        # S_matrix_path = './Data/S.mat'
        # S_data = scio.loadmat(S_matrix_path)
        # S_matrix = S_data['S']
        # # 将语义相似性与高斯相似性进行融合
        # dis_sim_matrix = (dis_gipsim_matrix+S_matrix)/2
        dis_sim_matrix = dis_gipsim_matrix
        circ_sim_matrix = circ_gipsim_matrix

        # 根据公式初步写一下
        # 先求两个相似矩阵的特征值和特征向量
        dis_eigenvalues, dis_eigenvectors = np.linalg.eig(dis_sim_matrix)
        circ_eigenvalues, circ_eigenvectors = np.linalg.eig(circ_sim_matrix)
        circ_diag_matrix = np.diag(circ_eigenvalues)
        dis_diag_matrix = np.diag(dis_eigenvalues)

        # 首先计算向量Z
        vectorz_part1 = np.kron(circ_diag_matrix, dis_diag_matrix)
        vectorz_part2 = np.kron(circ_diag_matrix, dis_diag_matrix) + 0.2 * np.eye(vectorz_part1.shape[0])
        #求vectorz_part2的逆运算
        vectorz_part2 = np.asarray(np.asmatrix(vectorz_part2).I)

        vectorz_part3 = np.dot(np.dot(dis_eigenvectors.T, rel_matrix.T), circ_eigenvectors)
        # 把vectorz_part3 的列向量堆叠起来
        for col in range(vectorz_part3.shape[1]):
            if col==0:
                temp_vector = vectorz_part3[:,col]
            else:
                temp_vector = np.concatenate((temp_vector, vectorz_part3[:,col]))
        vectorz_part3 = temp_vector

        vectorz = np.dot(np.dot(vectorz_part1, vectorz_part2), vectorz_part3)
        # # 把vectorz还原为89*533的矩阵
        # matrix_z = np.zeros((89,533))
        # for j in range(matrix_z.shape[1]):
        #     matrix_z[:,j] = vectorz[j*89:j*89+89]

        # matrix_z = np.zeros((62, 514))
        # for j in range(matrix_z.shape[1]):
        #     matrix_z[:, j] = vectorz[j * 62:j * 62 + 62]

        # matrix_z = np.zeros((40, 312))
        # for j in range(matrix_z.shape[1]):
        #     matrix_z[:, j] = vectorz[j * 40:j * 40 + 40]

        matrix_z = np.zeros((rel_matrix.shape[1], rel_matrix.shape[0]))
        for j in range(matrix_z.shape[1]):
            matrix_z[:, j] = vectorz[j * rel_matrix.shape[1]:j * rel_matrix.shape[1] + rel_matrix.shape[1]]

        prediction_matrix = np.dot(np.dot(circ_eigenvectors, matrix_z.T), dis_eigenvectors.T)

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
    mean_accuracy1 = np.mean(accuracy_arr)
    mean_recall = np.mean(np.mean(recall_arr, axis=1), axis=0)
    mean_precision = np.mean(np.mean(precision_arr, axis=1), axis=0)
    mean_F1 = np.mean(np.mean(F1_arr, axis=1), axis=0)
    print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f" % (mean_accuracy, mean_recall, mean_precision, mean_F1))

    roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)
    AUPR = np.trapz(mean_cross_precision, mean_cross_recall)
    print("AUC:%.4f,AUPR:%.4f" % (roc_auc, AUPR))

    # 存储tpr，fpr,recall,precision
    with h5py.File('./PlotFigure/DWNN-RLS_noncancer_5fold_AUC.h5','w') as hf:
        hf['fpr'] = mean_cross_fpr
        hf['tpr'] = mean_cross_tpr
    with h5py.File('./PlotFigure/DWNN-RLS_noncancer_5fold_AUPR.h5','w') as h:
        h['recall'] = mean_cross_recall
        h['precision'] = mean_cross_precision

    plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)
    # plt.savefig("./FinalResultPng/roc-DWNN-RLS_circad.png")
    print("runtime over, now is :")
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    plt.show()







