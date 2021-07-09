'''
基于拉普拉斯修正的朴素贝叶斯分类器
'''


import pandas as pd
import numpy as np
import copy

def create_attribute_set(data,d,n):
    
    att = [] #离散属性集
    for j in range(1,d-3):
        att1 = []
        for i in range(1,n):
            if data[i][j] not in att1:att1.append(data[i][j])
        att.append(att1)
    att = np.array(att,dtype = object)
    return att


def cal_pro(data,result,att):
    '''
    Calculate the set of Conditional probabilities & Priori probability
    After Laplaction correction
    '''
    pro_arr = np.ones(shape=(2,6,3))  #通道（好坏）2 属性数6 属性值3
    good_num = 0
    for i in range(len(result)):
        if result[i] == '是':good_num += 1
    bad_num = len(result)-good_num
    
    p1 = (good_num+1)/(len(data)+2)
    p2 = (bad_num+1)/(len(data)+2)   #先验概率

    for i in range(len(att)):
        for j in range(len(att[i])): 
            same_num1 = 0
            same_num2 = 0
            for k in range(len(data)):
                if data[k,i] == att[i][j] and result[k] == '是':same_num1 += 1   #第k行第i个属性 ?= 第i个属性的第j种属性值
                if data[k,i] == att[i][j] and result[k] == '否':same_num2 += 1
            pro_arr[0,i,j] = (same_num1+1) / (good_num+len(att[i]))
            pro_arr[1,i,j] = (same_num2+1) / (bad_num+len(att[i]))
    #print(pro_arr)
    return pro_arr,p1,p2


def test(test_arr,att,pro_arr,p1,p2):
    '''
    Calculate the predicted probability
    '''

    test_p1 = []
    test_p2 = []
    for i in range(len(att)):
        for j in range(len(att[i])):
            if att[i][j] in test_arr[0]:
                test_p1.append(pro_arr[0,i,j])
                test_p2.append(pro_arr[1,i,j])
    temp1,temp2 = 1,1
    for items in test_p1:temp1 = temp1 * items
    for items in test_p2:temp2 = temp2 * items
    pre_good = temp1*p1
    pre_bad = temp2*p2
    if pre_good >= pre_bad:
        predict = '好瓜'
    else:predict = '坏瓜' 
    return pre_good,pre_bad,predict
    

#main
data = pd.read_table('watermelon_data30.txt',delimiter = ',')
data =  np.array(data)
##print(data)
result = data[:,-1]
test_sample = [data[10]]
n,d = data.shape[0],data.shape[1]
att = create_attribute_set(data,d,n)
##print(att)
pro_arr,p1,p2 = cal_pro(data[:,1:7],result,att)
good_rate,bad_rate,predict= test(test_sample,att,pro_arr,p1,p2)
print('训练集为\n',data,'\n测试集为\n',test_sample,'\n机器学习方法为 基于拉普拉斯修正的朴素贝叶斯分类器')
print('\n好瓜概率为',good_rate,'\n坏瓜概率为',bad_rate,'\n\n>>预测为',predict)


