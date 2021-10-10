'''
@File: DMFCDA.py
@Author: Dong Yi
@Date: 2021/4/12 16:51
@Description: 
2021/4/12 这个是对模型DMFCDA的模型复现
其中关于深度学习部分，这里用tensorflow 2.0.0 中的eager excution模式
子模型形式 + keras 搭建深度学习框架
'''
import math
import random
from time import time

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_core.python.keras import regularizers
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sortscore



def DMFCDA(init):

    disease_input = tf.keras.Input(shape=(514,), name='disease_input')
    circRNA_input = tf.keras.Input(shape=(62,), name="circRNA_input")

    left = layers.Dense(514, name="left_dense_1", activation="relu", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(disease_input)
    left = layers.Dropout(0.005)(left)
    left = layers.Dense(257, name="left_dense_2", activation="relu", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(left)
    left = layers.Dropout(0.005)(left)
    left = layers.Dense(128, name="left_dense_3", activation="relu", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(left)

    right = layers.Dense(62, name="right_dense_1", activation="relu", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(circRNA_input)
    right = layers.Dropout(0.005)(right)
    right = layers.Dense(31, name="right_dense_2", activation="relu", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(right)
    right = layers.Dropout(0.005)(right)
    right = layers.Dense(15, name="right_dense_3", activation="relu", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(right)

    x = layers.concatenate([left, right], axis=1)

    final_vector = layers.Dense(143, name="final_dense_1", activation="relu", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(x)
    predict = layers.Dense(1, name="prediction_layer", activation="sigmoid", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(final_vector)

    model = tf.keras.Model(inputs=[disease_input, circRNA_input], outputs=predict)

    return model


def get_train_set(rel_matrix):
    circRNA_input, disease_input, label = [], [], []
    one_tuple = np.where(rel_matrix == 1)
    one_tuple_list = list(zip(one_tuple[0], one_tuple[1]))
    for (c,d) in one_tuple_list:
        # positive samples
        circRNA_input.append(rel_matrix[c,:])
        disease_input.append(rel_matrix[:,d])
        label.append(1)
        # negative samples
        j = np.random.randint(rel_matrix.shape[1])
        while (c,j) in one_tuple_list:
            j = np.random.randint(rel_matrix.shape[1])
        circRNA_input.append(rel_matrix[c,:])
        disease_input.append(rel_matrix[:,j])
        label.append(0)

    return circRNA_input, disease_input, label

def get_test_set(rel_matrix):
    circRNA_test_input, disease_test_input, label = [], [], []
    for i in range(rel_matrix.shape[0]):
        for j in range(rel_matrix.shape[1]):
            circRNA_test_input.append(rel_matrix[i,:])
            disease_test_input.append(rel_matrix[:,j])
            label.append(rel_matrix[i,j])
    return circRNA_test_input, disease_test_input, label

if __name__ == '__main__':
    num_negatives = 1
    epoches = 100
    batchsize = 100
    # 读取关系数据
    # with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]
    with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
        circrna_disease_matrix = hf['infor'][:]
    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]
    # with h5py.File('./Data/circ2Traits/circRNA_disease.h5', 'r') as hf:
    #       circrna_disease_matrix = hf['infor'][:]
    # with h5py.File('./Data/circad/circrna_disease.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    circrna_num = circrna_disease_matrix.shape[0]
    disease_num = circrna_disease_matrix.shape[1]

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
        init = tf.keras.initializers.TruncatedNormal(stddev=0.1)
        model = DMFCDA(init)
        model.compile(optimizer = tf.keras.optimizers.Adam(0.001),
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics = ['accuracy'])
        train_index = one_list[i:i + split]
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        for index in train_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        rel_matrix = new_circrna_disease_matrix

        for epoche in range(epoches):
            t1 = time()
            circRNA_input, disease_input, label = get_train_set(rel_matrix)

            model.fit([np.array(disease_input), np.array(circRNA_input)], np.array(label), epochs=1, batch_size=batchsize, verbose=1, shuffle=True)

            t2 = time()

        circ_test_input, dis_test_input, _ = get_test_set(rel_matrix)
        predictions = model.predict([np.array(dis_test_input), np.array(circ_test_input)], batch_size=100)

        prediction_matrix = np.zeros(rel_matrix.shape)
        for num in range(len(predictions)):
            row_num = num // disease_num
            col_num = num % disease_num
            prediction_matrix[row_num,col_num] = predictions[num][0]

        zero_matrix = np.zeros(prediction_matrix.shape).astype('int64')
        print(prediction_matrix.shape)

        prediction_matrix_temp = prediction_matrix.copy()
        prediction_matrix = prediction_matrix_temp + zero_matrix
        minvalue = np.min(prediction_matrix)
        prediction_matrix[np.where(roc_circrna_disease_matrix) == 2] = minvalue - 20
        sorted_circrna_disease_matrix, sorted_score_matrix = sortscore.sort_matrix(prediction_matrix,
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

    # # 存储tpr，fpr,recall,precision
    # with h5py.File('./PlotFigure/DMFCDA_circad_10fold_AUC.h5') as hf:
    #     hf['fpr'] = mean_cross_fpr
    #     hf['tpr'] = mean_cross_tpr
    # with h5py.File('./PlotFigure/DMFCDA_circad_10fold_AUPR.h5') as h:
    #     h['recall'] = mean_cross_recall
    #     h['precision'] = mean_cross_precision

    plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)
    print("runtime over, now is :")
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    plt.show()















        





