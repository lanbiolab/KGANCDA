'''
@Author: Dong Yi
@Date: 20201111
@Description:
这里首先尝试将存储的模型能否还原 -可以还原user和item的embedding向量
这里接着用自己的方式构造模型，并且注意关注loss函数该如何构建

'''
import math
import os
import h5py
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.regularizers import l2

import sortscore
from MakeSimilarityMatrix import MakeSimilarityMatrix
from Model.utility.parser import parse_args
import numpy as np
import matplotlib.pyplot as plt

# 定义计算circRNA前二十个相似circRNA的方法
def top_fif_sim_circ(circ_sim_matrix, u):

    top_circ_list = np.argsort(-circ_sim_matrix[u,:])
    top_fif_circ_list = top_circ_list[:10]
    top_fif_circ_list = top_fif_circ_list.tolist()
    if u in top_fif_circ_list:
        top_fif_circ_list = top_circ_list[:11]
        top_fif_circ_list = top_fif_circ_list.tolist()
        top_fif_circ_list.remove(u)

    return top_fif_circ_list

def find_associate_disease(top_fif_circ_list, relmatrix, associate_disease_set):
    for i in range(10):
        circ_id = top_fif_circ_list[i]
        for j in range(relmatrix.shape[1]):
            if relmatrix[circ_id,j]==1:
                associate_disease_set.add(j)

    return associate_disease_set

def init():
    return tf.keras.initializers.glorot_normal

def MyModel():
    inputs = tf.keras.Input(shape=(184,))
    dense1_tensor = Dense(46, kernel_regularizer=l2(0.0001), activation='relu', name='dense1')(inputs)
    dense1_out = Dropout(0.4)(dense1_tensor)
    dense2_tensor = Dense(12, kernel_regularizer=l2(0.0001), activation='relu', name='dense2')(dense1_out)
    dense2_out = Dropout(0.4)(dense2_tensor)
    # dense3_tensor = Dense(128, kernel_regularizer=l2(0),activation='relu', name='dense3')(dense2_out)
    # dense3_out = Dropout(0.4)(dense3_tensor)
    # dense4_tensor = Dense(32, kernel_regularizer=l2(0), activation='relu', name='dense4')(dense3_tensor)
    # dense4_out = Dropout(0.4)(dense4_tensor)
    prediciton_out = Dense(1, activation='sigmoid', kernel_initializer='glorot_normal', name='prediction')(dense2_out)

    model = tf.keras.Model(inputs = inputs, outputs = prediciton_out)

    return model

def get_train_instances(train_index, rel_matrix, circ_sim_matrix, circRNA_embedding, entity_embedding, disNum):
    # 获取训练集和测试集
    input_fusion_feature_x = []
    input_fusion_x_label = []
    for [u, i] in train_index:
        # with sess.as_default():
        #     kgat_circ_array = tf.nn.embedding_lookup(ua_embedding, u).eval()
        #     kgat_disease_array = tf.nn.embedding_lookup(entity_embedding, i).eval()
        # kgat_circ_array = entity_embedding[0][u,:]
        # kgat_disease_array = entity_embedding[0][i + rel_matrix.shape[0],:]
        kgat_circ_array = circRNA_embedding[u,:]
        kgat_cancer_array = entity_embedding[i,:]
        fusion_feature = np.concatenate((kgat_circ_array, kgat_cancer_array), axis=0)
        input_fusion_feature_x.append(fusion_feature.tolist())
        input_fusion_x_label.append(1)
        # 计算当前circRNA与其相似的circRNA前十五个
        top_fif_circ_list = top_fif_sim_circ(circ_sim_matrix, u)
        # 构建一个与这五个circRNA有关的集合
        associate_disease_set = set()
        # 找到这五个circRNA 与之有关的disease
        associate_disease_set = find_associate_disease(top_fif_circ_list, rel_matrix, associate_disease_set)
        # 负样本
        for num in range(4):
            j = np.random.randint(disNum)
            # while ([u, j] in train_index) or (j in associate_disease_set):
            #     j = np.random.randint(disNum)
            while ([u, j] in train_index):
                j = np.random.randint(disNum)
            # with sess.as_default():
            #     kgat_disease_array = tf.nn.embedding_lookup(entity_embedding, j).eval()
            # kgat_disease_array = entity_embedding[0][j + rel_matrix.shape[0],:]
            kgat_cancer_array = entity_embedding[j,:]
            fusion_feature = np.concatenate((kgat_circ_array, kgat_cancer_array), axis=0)
            input_fusion_feature_x.append(fusion_feature.tolist())
            input_fusion_x_label.append(0)

    return input_fusion_feature_x, input_fusion_x_label

def get_test_instances(rel_matrix, circRNA_embedding, entity_embedding):
    input_fusion_feature_test_x = []
    input_fusion_test_x_label = []
    # 测试集构造，同样每个正样本选择一个负样本作为测试集
    for row in range(rel_matrix.shape[0]):
        for col in range(rel_matrix.shape[1]):
            # with sess.as_default():
            #     kgat_circ_array = tf.nn.embedding_lookup(ua_embedding, row).eval()
            #     kgat_disease_array = tf.nn.embedding_lookup(entity_embedding, col).eval()
            # kgat_circ_array = entity_embedding[0][row, :]
            # kgat_disease_array = entity_embedding[0][col + rel_matrix.shape[0], :]
            kgat_circ_array = circRNA_embedding[row,:]
            kgat_cancer_array = entity_embedding[col,:]
            fusion_feature = np.concatenate((kgat_circ_array, kgat_cancer_array), axis=0)
            input_fusion_feature_test_x.append(fusion_feature.tolist())
            input_fusion_test_x_label.append(rel_matrix[row, col])

    return input_fusion_feature_test_x, input_fusion_test_x_label



if __name__ == '__main__':
    # 参数宏定义
    fold = 1

    # 首先读取circRNA-disease之间的联系
    with h5py.File('./Data/circRNA_cancer/circRNA_cancer_association.h5', 'r') as f:
        circrna_disease_matrix = f['infor'][:]
        circrna_disease_matrix_val = circrna_disease_matrix.copy()

    # 创建五折交叉运算后要记录的数据结构
    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []

    # 开始交叉验证
    for i in range(5):
        # 读取训练、测试下标
        with h5py.File('./Data/circRNA_cancer/circRNA_cancer_fold%d/circRNA_cancer_fold%d.h5' % (i+1, i+1), "r") as hs:
            train_index = hs['train_index'][:]
            test_index = hs['test_index'][:]

        train_index = train_index.tolist()
        test_index = test_index.tolist()

        # 读取user_embedding , entity_embedding
        config = tf.ConfigProto()
        sess = tf.Session(config= config)
        args = parse_args()
        model_type = 'kgat'
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        pretrain_path = './Model/weights/circRNA_cancer/circRNA_cancer_fold'+str(fold)+'/kgat_si_sum_kgat_l2/128-32/l0.0001_r1e-05-1e-05-0.01'
        saver = tf.train.import_meta_graph('./Model/weights/circRNA_cancer/circRNA_cancer_fold%d/kgat_si_sum_kgat_l2/128-32/l0.0001_r1e-05-1e-05-0.01/weights-19.meta'%fold)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            graph = tf.get_default_graph()
            entity_embedding = sess.run([graph.get_tensor_by_name('entity_embed:0')], feed_dict={})
            # ua_embedding, entity_embedding = sess.run([graph.get_tensor_by_name('user_embed:0'), graph.get_tensor_by_name('entity_embed:0')], feed_dict={})
            # circRNA_embedding, entity_embedding = sess.run(
            #     [graph.get_tensor_by_name('split:0'), graph.get_tensor_by_name('split:1')], feed_dict={
            #         model.node_dropout: eval(args.node_dropout),
            #         model.mess_dropout: eval(args.mess_dropout)
            #     })

        with h5py.File('./Data/large_circRNA_cancer/circRNA_cancer_fold%d/circRNA_cancer_32-16-8-4_embedding_fold%d_ADD.h5'%(fold, fold), 'r') as hf:
            circRNA_embedding = hf['circRNA_embedding'][:]
            ent_embedding = hf['entity_embedding'][:]

        # 抹除一部分关系
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        rel_matrix = new_circrna_disease_matrix

        # 记录circRNA，disease个数
        circNum = rel_matrix.shape[0]
        disNum = rel_matrix.shape[1]

        # # 计算当前已知关系矩阵的高斯相似性
        # makesimilaritymatrix = MakeSimilarityMatrix(rel_matrix)
        # circ_sim_matrix, dis_sim_matrix = makesimilaritymatrix.circsimmatrix, makesimilaritymatrix.dissimmatrix

        # with h5py.File('./Data/circRNA_cancer/circRNA_cancer_gipsim_matrix/circRNA_cancer_gip_sim_fold%d.h5'%fold, 'w') as hf:
        #     hf['circ_gipsim_matrix'] = circ_sim_matrix
        #     hf['dis_gipsim_matrix'] = dis_sim_matrix

        with h5py.File('./Data/circRNA_cancer/circRNA_cancer_gipsim_matrix/circRNA_cancer_gip_sim_fold%d.h5'%fold, 'r') as hf:
            circ_sim_matrix = hf['circ_gipsim_matrix'][:]
            dis_sim_matrix = hf['dis_gipsim_matrix'][:]

        fold += 1


        # 创建模型
        model = MyModel()
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        for epoch in range(60):
            # 获得训练集
            input_fusion_feature_x, input_fusion_x_label = get_train_instances(train_index, rel_matrix, circ_sim_matrix,
                                                                               circRNA_embedding, ent_embedding, disNum)
            history = model.fit(np.array(input_fusion_feature_x), np.array(input_fusion_x_label), epochs=1, batch_size=50)
        # 获得测试集
        input_fusion_feature_test_x, input_fusion_test_x_label = get_test_instances(rel_matrix, circRNA_embedding, ent_embedding)
        predictions = model.predict(np.array(input_fusion_feature_test_x), batch_size=50)

        prediction_matrix = np.zeros((rel_matrix.shape[0], rel_matrix.shape[1]))
        predictions_index = 0
        for row in range(prediction_matrix.shape[0]):
            for col in range(prediction_matrix.shape[1]):
                prediction_matrix[row, col] = predictions[predictions_index]
                predictions_index += 1

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

        # top_list = [10, 20, 50, 100, 200]
        # for num in top_list:
        #     P_matrix = sorted_circrna_disease_matrix[0:num, :]
        #     N_matrix = sorted_circrna_disease_matrix[num:sorted_circrna_disease_matrix.shape[0] + 1, :]
        #     top_count = np.sum(P_matrix == 1)
        #     print("top" + str(num) + ": " + str(top_count))

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
    print(roc_auc)
    print(AUPR)

    # 把tpr和fpr存起来
    with h5py.File('./results/KGANCDA_large_circRNA_cancer_4layer_ADD_32_AUC.h5') as hf:
        hf['fpr'] = mean_cross_fpr
        hf['tpr'] = mean_cross_tpr
    with h5py.File('./results/KGANCDA_large_circRNA_cancer_4layer_ADD_32_AUPR.h5') as f:
        f['recall'] = mean_cross_recall
        f['precision'] = mean_cross_precision

    plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)
    # plt.savefig("roc-gcn-fold5-13.png")
    print("runtime over, now is :")
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    plt.show()





# config = tf.ConfigProto()
# sess = tf.Session(config=config)
# args = parse_args()
# model_type = 'kgat'
# layer = '-'.join([str(l) for l in eval(args.layer_size)])
#
# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('./Model/weights/circRNA_cancer/circRNA_cancer_fold1/kgat_si_sum_kgat_l2/64-32/l0.0001_r1e-05-1e-05-0.01/weights-29.meta')
#     saver.restore(sess, tf.train.latest_checkpoint('./Model/weights/circRNA_cancer/circRNA_cancer_fold1/kgat_si_sum_kgat_l2/64-32/l0.0001_r1e-05-1e-05-0.01/'))
#     graph = tf.get_default_graph()
#
#     tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
#     for tensor_name in tensor_name_list:
#         print(tensor_name, '\n')
#
#     op_list = sess.graph.get_operations()
#     for op in op_list:
#         print(op.name)
#         print(op.values())
#






