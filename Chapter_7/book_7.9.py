import numpy as np

###已完成评价函数
###待完成迭代贝叶斯网 

    
def bic(data,result,m):
    '''
    calculate the num of score based on bayesian information criterion
    input:
    data   : dataset only
    result : result dataset
    att    : attribute 
    d      : demension of dataset
    n      : num of data
    m      : num of attribute in bayesian graph
    '''
    d,n = data.shape[1],data.shape[0]
    print(data,result,d,n,m)
    ###calculate att
    att = []
    for j in range(d):
        att1 = []
        for i in range(n):
            if data[i][j] not in att1:att1.append(data[i][j])
        att.append(att1)
    print(att)
    
    ###Calculate the set of log_likelihood & Priori probability
    ###After Laplaction correction
    p_arr = np.ones(shape=(2,d,3))  #通道（好坏）2 属性数6 属性值3
    good_num = 0
    for i in range(len(result)):
        if result[i] == '好瓜':good_num += 1
    bad_num = len(result)-good_num
    p1 = (good_num+1)/(len(data)+2)
    p2 = (bad_num+1)/(len(data)+2)   #先验概率
    for i in range(len(att)):
        for j in range(len(att[i])): 
            same_num1 = 0
            same_num2 = 0
            for k in range(len(data)):
                if data[k][i] == att[i][j] and result[k] == '好瓜':same_num1 += 1   #第k行第i个属性 ?= 第i个属性的第j种属性值
                if data[k][i] == att[i][j] and result[k] == '坏瓜':same_num2 += 1
            p_arr[0,i,j] = (same_num1+1) / (good_num+len(att[i]))          #Laplaction correction
            p_arr[1,i,j] = (same_num2+1) / (bad_num+len(att[i]))
    print('p_arr = ',p_arr)

    ###Calculate the likelihood and append
    likelihood_list = []
    for k in range(len(data)):
        p_good = []
        p_bad  = []
        for i in range(len(att)):
            for j in range(len(att[i])):
                if att[i][j] in data[k]:
                    p_good.append(p_arr[0,i,j])
                    p_bad.append(p_arr[1,i,j])
        temp1,temp2 = 1,1
        for items in p_good:temp1 = items*temp1
        for items in p_bad:temp2 = items*temp2
        pre_good = temp1*p1
        pre_bad = temp2*p2
        if pre_good >= pre_bad:likelihood_list.append(np.log(pre_good))
        elif pre_good < pre_bad:likelihood_list.append(np.log(pre_bad))
    likelihood_num = np.sum(likelihood_list)

    ###calculate the score
    score = 0.5*(np.log(m))*m + likelihood_num
    print('score = ',score)
    return score



data_20=[['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']]
labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感','类别']
data = np.array(data_20)
result = data[:,-1]
Beyesian_net_init = []
for i in range(7):Beyesian_net_init.append([j for j in range(7)])
print(Beyesian_net_init)
n,d = data.shape[0],data.shape[1]
bic_score = bic(data[:,1:6],result,m=7)