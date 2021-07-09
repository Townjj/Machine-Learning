'''
西瓜书习题8.3
基于决策树的Adaboost算法实现
2021/05/20-2021/05/29
'''


import numpy as np
import matplotlib.pyplot as plt
import math
import copy


def gini_index_continue(D,a,sample_weight):

    '''
    计算基尼值
    输入：数据集D，第a离散属性
    创建：candidate = [候选二分值]
          candidate_sorted = [排序后候选值(未二分)]
          D_discrete = [复制传入的数据集，并在遍历时用二分值替换连续值]
          can_gini_dict = {候选值：对应的基尼指数}
          min_candidate = 基尼指数最小的候选值
          temp_dict={属性值：[编号]}
          gini_index=(Dv/D)*gini(Dv)
          sample_weight:样本权重
        输出：基尼指数
    '''

    candidate = []
    candidate_sorted = [] 
    can_gini_dict = {}

    #计算候选值
    for item in D:
        candidate_sorted.append(item[a])
    candiadate_sorted = sorted(candidate_sorted)
    for i in range(len(D)-1):
        candidate.append((float(candiadate_sorted[i])+float(candiadate_sorted[i+1]))/2)

    for item in candidate:
        D_discrete = copy.deepcopy(D)                  #字符串列表一定要深复制，否则不独立
        for i in range(len(D)):
            if float(D[i][a]) > item:D_discrete[i][a] = 1
            else: (D_discrete[i][a]) = 0

        #根据属性值分类   
        temp_dict = {} ##{属性值:[编号]}
        for i in range(len(D_discrete)):
            if str(D_discrete[i][a]) in temp_dict.keys():  temp_dict[str(D_discrete[i][a])].append(int(D_discrete[i][0]))
            else: temp_dict[str(D_discrete[i][a])]=[int((D_discrete[i][0]))]

        #遍历计算基尼指数
        gini = 0
        weight_len = 0
        gini_index = 0
        for item1 in temp_dict.values():
            p1_num ,p2_num = 0,0
            for index in item1:
                if data[index-1][-1] == '好瓜': p1_num += 1-sample_weight[index-1]
                if data[index-1][-1] == '坏瓜': p2_num += 1-sample_weight[index-1]
                weight_len += sample_weight[index-1]
            gini = 1-(p1_num/len(item1))**2-(p2_num/len(item1))**2
            gini_index += (len(item1) / len(D))*gini
        can_gini_dict[str(item)] = gini_index
        #print('\n69',temp_dict)
    print(can_gini_dict)

    #选出最小基尼值的二分属性值
    for key,value in can_gini_dict.items():
        if (value == min(can_gini_dict.values())):min_candidate = key
    for i in range(len(D)):
            if float(D[i][a]) > float(min_candidate): 
                if a==1:D_discrete[i][a] = '密度>'+str(min_candidate)
                else:D_discrete[i][a] = '含糖量>'+str(min_candidate)
            elif a == 1:D_discrete[i][a] = '密度<'+str(min_candidate)
            elif a == 2:D_discrete[i][a] = '含糖量<'+str(min_candidate)
    #print(can_gini_dict)
    return(min_candidate,min(can_gini_dict.values()),D_discrete)

def treegenerate(D,A,d,sample_weight):
    '''
    生成决策树
    输入：D:数据集
          A:属性集
        lever:{好瓜：编号}
        sample_weight:样本权重
    '''
    data_init = copy.deepcopy(data)
    lever = {}
    #A为空集
    if not A :
        predict_true_list = []
        lever = {'好瓜':[ ],'坏瓜':[ ]}
        for i in range(0,len(D)):
            if str(D[i][-1]) == '好瓜':lever['好瓜'].append(int(D[i][0]))
            if str(D[i][-1]) == '坏瓜':lever['坏瓜'].append(int(D[i][0]))
        if len(lever['好瓜'])  > len(lever['坏瓜']):
            lever['好瓜'] += lever['坏瓜']
            del lever['坏瓜']
            for index in range(len(lever['好瓜'])):
                if D[index-1][-1] == '好瓜':predict_true_list.append(D[index-1][0])
        else:
            lever['坏瓜'] += lever['好瓜']
            del lever['好瓜']
            for index in range(len(lever['坏瓜'])):
                if D[index-1][-1] == '坏瓜':predict_true_list.append(D[index-1][0])
        #print('A为空集  lever = ',lever,predict_true_list)

        return lever,predict_true_list

    # D在中类别一致
    predict_true_list = []
    lever = {'好瓜':[ ],'坏瓜':[ ]}
    for a in A:
        for i in range(0,len(D)):
            if D[i][-1] == D[i-1][-1]:
                if str(D[i][-1]) == '好瓜':lever['好瓜'].append(int(D[i][0]))
                if str(D[i][-1]) == '坏瓜':lever['坏瓜'].append(int(D[i][0]))
            else:
                lever = {}
                break
        if lever == {}:break
        if len(lever['好瓜'])  > len(lever['坏瓜']):
            del lever['坏瓜']
            for index in range(len(lever['好瓜'])):
                if D[index-1][-1] == '好瓜':predict_true_list.append(D[index-1][-1])
        else:
            del lever['好瓜']
            for index in range(len(lever['坏瓜'])):
                if D[index-1][-1] == '坏瓜':predict_true_list.append(D[index-1][-1])
        #print('D中类别一致　lever = ',lever,predict_true_list)
        return lever,predict_true_list
    #D在A中属性一致
    lever = {'好瓜':[ ],'坏瓜':[ ]}
    predict_true_list = []
    for a in A:
        for i in range(0,len(D)):
            if D[i][a] == D[-1][a]:
                if str(D[i][-1]) == '好瓜':lever['好瓜'].append(int(D[i][0]))
                if str(D[i][-1]) == '坏瓜':lever['坏瓜'].append(int(D[i][0]))
            else:
                lever = {}
                break
        if lever == {}:break
        if len(lever['好瓜'])  > len(lever['坏瓜']):
            lever['好瓜'] += lever['坏瓜']
            del lever['坏瓜']
            for index in range(len(lever['好瓜'])):
                if D[index-1][-1] == '好瓜':predict_true_list.append(D[index-1][-1])
        else:
            lever['坏瓜'] += lever['好瓜']
            del lever['好瓜']
            for index in range(len(lever['坏瓜'])):
                if D[index-1][-1] == '坏瓜':predict_true_list.append(D[index-1][-1])
        #print('D在A中属性一致 lever = ',lever,predict_true_list)
        return lever,predict_true_list
    
    #其他情况选取最优划分属性
    max_a = 0
    gini_init = 1000
    gini_list = {}  #{属性：gini}
    for a in A:
        max_t,min_gini_index,Da_dis = gini_index_continue(D,a,sample_weight)
        for i in range(0,len(D)):D[i][a] = Da_dis[i][a]
        gini_list[str(d[0][a])] = min_gini_index
        #取出最小的基尼指数作为分类属性
        if min_gini_index < gini_init :
            max_gini = min_gini_index
            max_a = a
        print(gini_list)
    
    #根据属性值分类{属性值：编号}
    tree = {}
    for i in range(0,len(D)):
        if str(D[i][max_a]) in tree.keys():     
            tree[str(D[i][max_a])].append(int(D[i][0]))
        else: tree[str(D[i][max_a])]=[int((D[i][0]))]
    #print('tree = ',tree)
    A.remove(max_a)
    #print('159tree',tree)
    for key in tree.keys():
        Dv_temp= []
        for i in range(0,len(tree[key])):
            Dv_temp.append(data_init[(tree[key])[i]-1])
        Dv = copy.deepcopy(Dv_temp)
        #print(Dv_temp)
        print('\n194\n--------迭代信息--------','\nDv = ',Dv,'\nA = ',A,'\n--------进入迭代--------')
        tree[key],predict_true_list_return = treegenerate(Dv,A,d,sample_weight=sample_weight)
        predict_true_list += predict_true_list_return
    #print('187tree_rate=',len(predict_true_list)/len(data))
    return tree,predict_true_list

def adaboost(data,result,T):
    '''
    输入:训练数据data
            数据属性序号列表A
            数据集表头d
            训练轮数(基学习器个数)T
    '''
    sample_weight = [1/(len(data))] * len(data) #样本权值
    predict_list = np.zeros((T,len(data)))      #T个基学习器学习的Tx17个结果
    alpha = [1] * T                             #基学习器权值
    for t in range(T):
        predict_list[t]  = [-1] * len(data)
        train_data = copy.deepcopy(data)
        tree,predict_true_list= treegenerate(train_data,A=[1,2],d=[['编号','密度','含糖量','好瓜']],sample_weight=sample_weight)
        predict_true_rate = len(predict_true_list) / len(data)
        predict_list[t,:] = result
        for index in range(len(data)):
            if index+1 not in predict_true_list:
                predict_list[t,index] = result[index]*(-1)

        print(tree,'\n',predict_true_list)
        if predict_true_rate < 0.5 :break
        alpha[t] = 0.5*math.log((predict_true_rate)/(1-predict_true_rate))
        for i in range(len(data)): 
            sample_weight[i] = sample_weight[i] * math.exp(-alpha[t]*result[i]*predict_list[t,i])
        print('209sample_weight= ',t,sample_weight)
        sample_weight = sample_weight / np.sum(sample_weight)
    alpha = alpha / np.sum(alpha)
    fusion_result = np.array(alpha).reshape(1,T) @ predict_list #1*TxT*17 = 1*17
    fusion_result = np.sum(fusion_result,axis=0)
    fusion_true_num = 0 
    for i in range(len(data)):
        if fusion_result[i] > 0 : 
            fusion_result[i] = '1'
            if data[i][-1] == '好瓜':
                fusion_true_num += 1
        else:
            fusion_result[i] = '-1'
            if data[i][-1] == '坏瓜':
                fusion_true_num += 1
    fusion_true_rate = fusion_true_num / len(data)
    return fusion_result,fusion_true_rate,alpha

## Main func
data = [[1,0.697, 0.460, 1],
        [2,0.774, 0.376, 1],
        [3,0.634, 0.264, 1],
        [4,0.608, 0.318, 1],
        [5,0.556, 0.215, 1],
        [6,0.403, 0.237, 1],
        [7,0.481, 0.149, 1],
        [8,0.437, 0.211, 1],
        [9,0.666, 0.091, -1],
        [10,0.243, 0.267, -1],
        [11,0.245, 0.057, -1],
        [12,0.343, 0.099, -1],
        [13,0.639, 0.161, -1],
        [14,0.657, 0.198, -1],
        [15,0.360, 0.370, -1],
        [16,0.593, 0.042, -1],
        [17,0.719, 0.103, -1]]
result=[]
for i in range(len(data)):result.append(data[i][-1])
for i in range(len(data)):
    if data[i][-1] == 1:data[i][-1]='好瓜'
    else:data[i][-1] = '坏瓜'
tree_train_data = copy.deepcopy(data)

T = 5
ada_boost_result,ada_boost_rate,alpha = adaboost(data,result,T)
print('\n---Adaboost---\n',T,'个基学习器权值为\n',alpha)
print('集成学习器输出结果为',ada_boost_result,'\n正确率为',ada_boost_rate,'\n------------')

tree,predict_true_list= treegenerate(tree_train_data,A=[1,2],d=[['编号','密度','含糖量','好瓜']],sample_weight=[1/(len(data))] * len(data))
print('---单个决策树的正确率为---\n',tree,len(predict_true_list)/len(data),'\n-----------\n')