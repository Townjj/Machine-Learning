import numpy as np

import math
import copy


def entro(D):
    '''
    计算信息熵
    输入：数据集D
    输出：数据集D的信息熵
    '''
    a_num = 0
    b_num = 0
    a = 0
    b = 0
    result = 0
    for item in D:
        if item[-1] == '好瓜': a_num += 1
        if item[-1] == '坏瓜': b_num += 1
    a = a_num/len(D)
    b = b_num/len(D)
    if a_num == 0 and b_num == 0: result = 0
    if a_num == 0: result = -(b*math.log(b,2))
    if b_num == 0: result = -(a*math.log(a,2))
    if a_num != 0 and b_num !=0 : result = -(a*math.log(a,2)+b*math.log(b,2))
    #print('Return of entro function = ',result )
    return result


def gain_discrete(D,a):
    '''
    计算离散属性值的信息增益
    输入：数据集D，第a离散属性
    创建：temp={属性值：[编号]}
          temp1=(Dv/D)*ent(Dv)
          Dv=在第a离散属性上取值为av的集合
    输出：信息增益
    '''
    temp1 = 0
    temp = {}
    #根据属性值分类
    for i in range(0,len(D)):
        if str(D[i][a]) in temp.keys():     
            temp[str(D[i][a])].append(int(D[i][0]))
        else: temp[str(D[i][a])]=[int((D[i][0]))]
    #print('Classify dict = ',temp)
    #遍历计算信息增益
    for item in temp.values():
        Dv = []
        for i in range(0,len(item)):
            Dv.append(data[item[i]])
        temp1 += (len(item)/len(D))*entro(Dv)
    #print('Classify =',data[0][a],' Return of gain_discrete function = ',entro(data[1:18])-temp1)
    return(entro(data[1:18])-temp1)


def gain_continue(D,a):
    '''
    计算离散属性值的信息增益
    输入：数据集D，第a离散属性
    创建：temp = [候选二分值]
          temp_list_sorted = [排序后候选值]
          Da_dis = [复制传入的数据集，并在遍历时用二分值替换连续值]
          t = {候选值：对应的信息增益}
          max_t = 信息增益最大的候选值
    输出：t,max_t
    '''
    temp = []
    temp_list_sorted = [] 
    t = {}
    #计算候选值
    for item in D:
        temp_list_sorted.append(item[a])
    temp_list_sorted = sorted(temp_list_sorted)
    for i in range(0,len(D)-1):
        temp.append((float(temp_list_sorted[i])+float(temp_list_sorted[i+1]))/2)
    Da_dis = []
    for item in temp:
        Da_dis = copy.deepcopy(D)                  #字符串列表一定要深复制，否则不独立
        for i in range(0,len(Da_dis)):
            if float(D[i][a]) > item: Da_dis[i][a] = 1
            else: (Da_dis[i][a]) = 0
        t[str(item)] = gain_discrete(Da_dis,a)
    for key,value in t.items():
        if (value == max(t.values())):max_t = key
    #print('候选值增益为 : ',t,'\n信息增益最大的候选值为：',max_t)
    #输出最大候选值情况下的Da_dis
    for i in range(0,len(Da_dis)):
            if float(D[i][a]) > float(max_t): 
                if a==7:Da_dis[i][a] = '密度>'+str(max_t)
                else:Da_dis[i][a] = '含糖量>'+str(max_t)
            elif a == 7:Da_dis[i][a] = '密度<'+str(max_t)
            elif a == 8:Da_dis[i][a] = '含糖量<'+str(max_t)

    return(max(t.values()),Da_dis)


def treegenerate(D,A):
    '''
    生成决策树
    输入：D:数据集
          A:属性集
        lever:{好瓜：编号}
    '''

    lever = {}
    #A为空集
    if not A :
        lever = {'好瓜':[ ],'坏瓜':[ ]}
        for i in range(0,len(D)):
            if str(D[i][-1]) == '好瓜':lever['好瓜'].append(int(D[i][0]))
            if str(D[i][-1]) == '坏瓜':lever['坏瓜'].append(int(D[i][0]))
        if len(lever['好瓜'])  > len(lever['坏瓜']):
            lever['好瓜'] += lever['坏瓜']
            del lever['坏瓜']
        else:
            lever['坏瓜'] += lever['好瓜']
            del lever['好瓜']
        print('A为空集  lever = ',lever)
        return lever

    # D在中类别一致
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
        if len(lever['好瓜'])  > len(lever['坏瓜']):del lever['坏瓜']
        else:del lever['好瓜']
        print('D中类别一致　lever = ',lever)
        return lever

    #D在A中属性一致
    lever = {'好瓜':[ ],'坏瓜':[ ]}
    for a in A:
        for i in range(0,len(D)):
            if D[i][a] == D[-1][a]:
                if str(D[i][-1]) == '是':lever['好瓜'].append(int(D[i][0]))
                if str(D[i][-1]) == '否':lever['坏瓜'].append(int(D[i][0]))
            else:
                lever = {}
                break
        if lever == {}:break
        if len(lever['好瓜'])  > len(lever['坏瓜']):
            lever['好瓜'] += lever['坏瓜']
            del lever['坏瓜']
        else:
            lever['坏瓜'] += lever['好瓜']
            del lever['好瓜']
        print('D在A中属性一致 lever = ',lever)
        return lever
    
    #其他情况选取最优划分属性
    max_a = 0
    max_gain = 0
    max_gain1 = 0
    max_gain2 = 0
    gain_list = {}  #{属性：gain}
    for a in A:
        if a < 7 : 
            max_gain1 = gain_discrete(D,a)
            gain_list[str(data[0][a])] = max_gain1
        if a == 7 or a == 8:
            #print('170',len(D),D,a) 
            max_gain2,Da_dis = gain_continue(D,a)
            for i in range(0,len(D)):D[i][a] = Da_dis[i][a] 
            gain_list[str(D[0][a])] = max_gain2
            
        if max_gain1 > max_gain:
            max_gain = max_gain1
            max_a = a
        if max_gain2 > max_gain:
            max_gain = max_gain2
            max_a = a
    
    tree = {}
    #根据属性值分类{属性值：编号}
    for i in range(0,len(D)):
        if str(D[i][max_a]) in tree.keys():     
            tree[str(D[i][max_a])].append(int(D[i][0]))
        else: tree[str(D[i][max_a])]=[int((D[i][0]))]
    #print('tree = ',tree)
    A.remove(max_a)
    
    for key in tree.keys():
        Dv = []
        for i in range(0,len(tree[str(key)])): Dv.append(data[(tree[key])[i]])
        #print('\n--------迭代信息--------','\nDv = ',Dv,'\nA = ',A,'\n--------进入迭代--------')
        tree[key] = treegenerate(Dv,A)
        
    print('tree = ',tree)
    return tree


def find_list(tree):
    '''
    输入决策树，返回好瓜坏瓜类别,递归
    '''
    global GOOD 
    global BAD
    for key,value in tree.items():
        if key == '好瓜':
            GOOD += value
            return GOOD,BAD
        if key == '坏瓜':  
            BAD += value
            return GOOD,BAD
        else: GOOD,BAD = find_list(value)
    print('3',GOOD,BAD)
    return GOOD,BAD


def true_rate(tree):
    '''
    输入决策树，计算预测的正确率
    '''
    count = 0
    good_predict_list,bad_predict_list = find_list(tree)
    for item in good_predict_list:
        if data[int(item)][-1] == '好瓜': count += 1
    for item in bad_predict_list:
        if data[int(item)][-1] == '坏瓜': count += 1
    true_rate = count*100/(len(bad_predict_list)+len(good_predict_list))
    print('正确率为 ',true_rate,'%')
    return true_rate


#读取数据集3.0并整理
filename = 'D:\OneDrive同步\OneDrive\\2021\Code_test\watermelon_data30.txt'
with open(filename,encoding='utf-8') as file_object:
    lines = file_object.readlines()
    data_lines = []
    for line in lines:
        data_lines.append(line.rstrip())
data=[[0 for _ in range(10)] for _ in range(18)]
for i in range(0,len(data_lines)):
    temp = ''
    j = 0
    for each in data_lines[i]:
        if each == ',':
            data[i][j] = temp
            j += 1
            temp=''
        elif each == data_lines[i][-1]:
            temp += each
            data[i][-1] = temp
        else:temp += each
for item in data:
    if item[-1] == '是':item[-1] = '好瓜'
    elif item[-1] == '否':item[-1] = '坏瓜'

D = copy.deepcopy(data)
D = (D[1:18])
A = [1,2,3,4,5,6,7,8]

tree = treegenerate(D,A)
print('------Result-------','\ntree=',tree)

GOOD  = []
BAD = []
print(true_rate(tree))
