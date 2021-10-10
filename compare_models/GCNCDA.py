'''
@Author: Dong Yi
@Date: 2020.12.23
@Description: 这是对GCNCDA的python实现
原本是Wang lei用matlab实现的版本
2021/05/22 这里把数据换成了circRNA-cancer之间的关系
'''
import math
import os
import pickle
import random

import h5py
import numpy as np
from scipy import sparse

from MakeSimilarityMatrix import MakeSimilarityMatrix
from scipy.io import loadmat

if __name__ == '__main__':
    # # 读取关系数据
    # with h5py.File('./Data/circR2cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]
    #     circrna_disease_matrix_val = circrna_disease_matrix.copy()

    with h5py.File('./Data/circR2Noncancer/circRNA_disease_matrix.h5', 'r') as hf:
        circrna_disease_matrix = hf['infor'][:]
        circrna_disease_matrix_val = circrna_disease_matrix.copy()

    # # 这里读取one_list
    # with h5py.File('./Data/circR2Disease/one_list_file/one_list.h5', 'r') as f:
    #     one_list = f['one_list'][:]

    # 数据被分为5分，这里只需要一份的长度
    index_tuple = np.where(circrna_disease_matrix)
    one_list = list(zip(index_tuple[0], index_tuple[1]))

    temp_list = []
    for temp in one_list:
        temp_list.append(tuple(temp))
    one_list = temp_list
    split = math.ceil(len(one_list) / 5)

    # 创建五折交叉运算后要记录的数据结构
    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []
    fold = 1

    for i in range(0, len(one_list), split):
        # 把一部分已知关系置零
        # test_index = one_list[i:i + split]
        # train_index = list(set(one_list) - set(test_index))

        with h5py.File('./Data/circR2NonCancer/circ_dis_fold{}/circRNA_disease_fold{}.h5'.format(fold, fold), 'r') as hf:
            train_index = hf['train_index'][:]
            test_index = hf['test_index'][:]
        train_index = train_index.tolist()
        test_index = test_index.tolist()

        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        rel_matrix = new_circrna_disease_matrix

        # # 计算当前已知关系矩阵的高斯相似性
        # makesimilaritymatrix = MakeSimilarityMatrix(rel_matrix)
        # circ_gipsim_matrix, dis_gipsim_matrix = makesimilaritymatrix.circsimmatrix, makesimilaritymatrix.dissimmatrix

        # with h5py.File('./Data/circR2cancer/circ_can_fold{}/GIP_sim_fold{}.h5'.format(fold, fold), 'w') as hf:
        #     hf['circ_gipsim_matrix'] = circ_gipsim_matrix
        #     hf['dis_gipsim_matrix'] = dis_gipsim_matrix

        with h5py.File('./Data/circR2NonCancer/circRNA_disease_gipsim_matrix/circRNA_disease_gip_sim_fold{}.h5'.format(fold), 'r') as hf:
            circ_gipsim_matrix = hf['circ_gipsim_matrix'][:]
            dis_gipsim_matrix = hf['dis_gipsim_matrix'][:]

        # with h5py.File('./Data/circR2Disease/GIP_sim_matrix.h5', 'r') as hf:
        #     circ_gipsim_matrix = hf['circ_gipsim_matrix'][:]
        #     dis_gipsim_matrix = hf['dis_gipsim_matrix'][:]

        # # 加载疾病的相似性
        # SV1_matrix_dict = loadmat('./Data/circR2Disease/disease_file/S.mat')
        # SV1_matrix = SV1_matrix_dict['S']
        # SV2_matrix_dict = loadmat('./Data/circR2Disease/disease_file/SS.mat')
        # SV2_matrix = SV2_matrix_dict['SS']
        # dis_sem_matrix = (SV1_matrix + SV2_matrix) / 2
        #
        # Dsim = np.zeros((rel_matrix.shape[1], rel_matrix.shape[1]))
        #
        # for m in range(Dsim.shape[0]):
        #     for n in range(Dsim.shape[1]):
        #         if(dis_sem_matrix[m,n]==0):
        #             Dsim[m,n]= dis_gipsim_matrix[m,n]
        #         else:
        #             Dsim[m,n] = dis_sem_matrix[m,n]

        Rsim = circ_gipsim_matrix.copy()
        Dsim = dis_gipsim_matrix.copy()

        # 开始凑训练样本的 FV 和 标签
        FV_train = []
        FV_train_label = []
        final_train_index = []
        negative_sample = []
        for (c,d) in train_index:
            # 正样本
            temp_feature = []
            Rsim_list = (Rsim[c, :]).tolist()
            Dsim_list = (Dsim[d,:]).tolist()
            temp_feature = Rsim_list + Dsim_list
            FV_train.append(temp_feature)
            FV_train_label.append([1,0])
            final_train_index.append((c,d))
        # 凑负样本的FV 和 标签
        for num in range(len(train_index)):
            c = np.random.randint(0,rel_matrix.shape[0])
            d = np.random.randint(0,rel_matrix.shape[1])
            while ((c,d) in train_index) or ((c,d) in negative_sample) or ((c,d) in final_train_index) :
                c = np.random.randint(0,rel_matrix.shape[0])
                d = np.random.randint(0,rel_matrix.shape[1])
            Rsim_list = (Rsim[c, :]).tolist()
            Dsim_list = (Dsim[d, :]).tolist()
            temp_feature = Rsim_list + Dsim_list
            FV_train.append(temp_feature)
            FV_train_label.append([0,1])
            negative_sample.append((c,d))
            final_train_index.append((c,d))

        final_train_index = list(set(final_train_index))

        # 开始凑测试集
        FV_test = []
        FV_test_label = []
        for row in range(rel_matrix.shape[0]):
            for col in range(rel_matrix.shape[1]):
                if (row,col) not in final_train_index:
                    temp_feature = []
                    Rsim_list = (Rsim[row, :]).tolist()
                    Dsim_list = (Dsim[col, :]).tolist()
                    temp_feature = Rsim_list + Dsim_list
                    FV_test.append(temp_feature)
                    if circrna_disease_matrix[row,col]==1:
                        FV_test_label.append([1,0])
                    else:
                        FV_test_label.append([0,1])

        # 开始构造graph
        graph = {}
        for m in range(rel_matrix.shape[0]):
            for n in range(rel_matrix.shape[1]):
                graph_related_list = []
                related_circ_id = []
                related_dis_id = []
                # 找这个circRNA与其他哪些疾病有关
                related_dis = np.where(rel_matrix[m,:]==1)
                related_dis_id = related_dis[0].tolist()
                # 找这个疾病与其他哪些circRNA有关
                related_circ = np.where(rel_matrix[:,n]==1)
                related_circ_id = related_circ[0].tolist()
                for h in related_dis_id:
                    if h != n:
                        graph_related_list.append(rel_matrix.shape[1] * m + h)
                for f in related_circ_id:
                    if f != m:
                        graph_related_list.append(rel_matrix.shape[1] * f + n)
                graph[rel_matrix.shape[1] * m + n] = graph_related_list

        # 整合为供给FastGCN的数据形式，用pickle打包好
        allx = sparse.csr_matrix(FV_train)
        ally = FV_train_label.copy()
        tx = sparse.csr_matrix(FV_test)
        ty = FV_test_label.copy()
        x = sparse.csr_matrix(FV_train)
        y = FV_train_label.copy()

        # 将测试集写入一个文件
        with open('./Data/circR2NonCancer/circ_dis_fold{}/ind.circRNA_disease.test.index'.format(fold),'w') as hf:
            for m in range(rel_matrix.shape[0]):
                for n in range(rel_matrix.shape[1]):
                    if (m,n) not in final_train_index:
                        hf.write(str(m * rel_matrix.shape[1] + n)+'\n')

        # 做成文件
        with open('./Data/circR2NonCancer/circ_dis_fold{}/ind.circRNA_disease.x'.format(fold), 'wb') as hf:
            pickle.dump(x,hf)
            hf.close()
        with open('./Data/circR2NonCancer/circ_dis_fold{}/ind.circRNA_disease.y'.format(fold), 'wb') as hf:
            pickle.dump(y,hf)
            hf.close()
        with open('./Data/circR2NonCancer/circ_dis_fold{}/ind.circRNA_disease.tx'.format(fold), 'wb') as hf:
            pickle.dump(tx,hf)
            hf.close()
        with open('./Data/circR2NonCancer/circ_dis_fold{}/ind.circRNA_disease.ty'.format(fold), 'wb') as hf:
            pickle.dump(ty,hf)
            hf.close()
        with open('./Data/circR2NonCancer/circ_dis_fold{}/ind.circRNA_disease.allx'.format(fold), 'wb') as hf:
            pickle.dump(allx,hf)
            hf.close()
        with open('./Data/circR2NonCancer/circ_dis_fold{}/ind.circRNA_disease.ally'.format(fold), 'wb') as hf:
            pickle.dump(ally,hf)
            hf.close()
        with open('./Data/circR2NonCancer/circ_dis_fold{}/ind.circRNA_disease.graph'.format(fold), 'wb') as hf:
            pickle.dump(graph, hf)
            hf.close()

        # 把负样本存储起来
        with h5py.File('./Data/circR2Noncancer/circ_dis_fold{}/negative_sample.h5'.format(fold), 'w') as f:
            f['negative_sample'] = negative_sample

        fold += 1































