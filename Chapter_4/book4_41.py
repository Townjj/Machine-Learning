import numpy as np
import matplotlib.pyplot as plt
import math
import copy


def gini(D):
    '''
    计算基尼值
    输入：数据集D
    输出：数据集D的基尼值
    '''
    p1_num = 0
    p2_num = 0
    p1 = 0
    p2 = 0
    gini = 0
    for item in D:
        if item[-1] == '好瓜': p1_num += 1
        if item[-1] == '坏瓜': p2_num += 1
    p1 = p1_num/len(D)
    p2 = p2_num/len(D)
    gini = 1-p1*p1-p2*p2
    #print('Return of gini function = ', gini )
    return gini


def gini_index_discrete(D,a):
    '''
    计算离散属性值的基尼指数
    输入：数据集D，第a离散属性
    创建：temp={属性值：[编号]}
          gini_index=(Dv/D)*gini(Dv)
          Dv=在第a离散属性上取值为av的集合
    输出：基尼指数
    '''
    gini_index = 0
    temp = {}
    #根据属性值分类
    for i in range(0,len(D)):
        if str(D[i][a]) in temp.keys():     
            temp[str(D[i][a])].append(int(D[i][0]))
        else: temp[str(D[i][a])]=[int((D[i][0]))]
    #print('Classify dict = ',temp)
    #遍历计算基尼指数
    for item in temp.values():
        Dv = []
        for i in range(0,len(item)):
            Dv.append(data[item[i]])
        gini_index += (len(item)/len(D))*gini(Dv)
    #print('Classify =',data[0][a],' Return of gini_discrete function = ',gini_index)
    return gini_index


def gini_index_continue(D,a):
    '''
    计算离散属性值的基尼指数
    输入：数据集D，第a离散属性
    创建：temp = [候选二分值]
          temp_list_sorted = [排序后候选值]
          Da_dis = [复制传入的数据集，并在遍历时用二分值替换连续值]
          t = {候选值：对应的基尼指数}
          max_t = 基尼指数最大的候选值
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
        for i in range(0,len(D)):
            if float(D[i][a]) > item: Da_dis[i][a] = 1
            else: (Da_dis[i][a]) = 0
        t[str(item)] = gini_index_discrete(Da_dis,a)
    for key,value in t.items():
        if (value == max(t.values())):max_t = key
    #print('候选值增益为 : ',len(t),t,'\n基尼指数最大的候选值为：',max_t)
    #输出最大候选值情况下的Da_dis
    for i in range(0,len(D)):
            if float(D[i][a]) > float(max_t): 
                if a==7:Da_dis[i][a] = '密度>'+str(max_t)
                else:Da_dis[i][a] = '含糖量>'+str(max_t)
            elif a == 7:Da_dis[i][a] = '密度<'+str(max_t)
            elif a == 8:Da_dis[i][a] = '含糖量<'+str(max_t)

    return(max_t,max(t.values()),Da_dis)


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
        #print('A为空集  lever = ',lever)
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
        #print('D中类别一致　lever = ',lever)
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
        #print('D在A中属性一致 lever = ',lever)
        return lever
    
    #其他情况选取最优划分属性
    min_a = 0
    min_gini = 1
    min_gini1 = 1
    min_gini2 = 1
    gini_list = {}  #{属性：gini}
    for a in A:
        if a < 7 : 
            min_gini1 = gini_index_discrete(D,a)
            gini_list[str(d[0][a])] = min_gini1
        if a == 7 or a == 8:
            max_t,min_gini2,Da_dis = gini_index_continue(D,a)
            for i in range(0,len(D)):D[i][a] = Da_dis[i][a]
            gini_list[str(d[0][a])] = min_gini2
        if min_gini1 < min_gini :
            min_gini = min_gini1
            min_a = a
        if min_gini2 < min_gini :
            min_gini = min_gini2
            min_a = a
    #print('177',gini_list,str(d[0][min_a]))
    
    #根据属性值分类{属性值：编号}
    tree = {}
    global feature_dict
    true_count_temp,false_count_temp = 0,0
    for j in range(0,len(feature_dict[str(min_a)])):
        tree[(feature_dict[str(min_a)])[j]] = []
    for i in range(0,len(D)):
        if str(D[i][min_a]) in tree.keys(): 
            tree[str(D[i][min_a])].append(int(D[i][0]))
        if D[i][-1] == '好瓜':true_count_temp += 1
        elif D[i][-1] == '坏瓜':false_count_temp += 1
    #空属性值处理
    for key,value in tree.items():
        if value == [] :
            if true_count_temp >= false_count_temp: 
                tree[str(key)] = {'好瓜':[]}
            else:tree[str(key)] = {'坏瓜':[]}
    A.remove(min_a)
    for key in tree.keys():
        Dv_temp= []
        if isinstance(tree[str(key)] ,list):
            for i in range(0,len(tree[str(key)])):
                Dv_temp.append(d[(tree[key])[i]])
                Dv = copy.deepcopy(Dv_temp)
            #print('\n194\n--------迭代信息--------','\nDv = ',Dv,'\nA = ',A,'\n--------进入迭代--------')
            tree[key] = treegenerate(Dv,A)
    return tree


def find_list1(tree):
    '''
    输入决策树，返回好瓜坏瓜类别,递归
    '''
    global GOOD,BAD
    
    for key,value in tree.items():
        if key == '好瓜':
            GOOD += value
            return GOOD,BAD
        if key == '坏瓜':  
            BAD += value
            return GOOD,BAD
        else: GOOD,BAD = find_list1(value)
    #print('good=',GOOD,'\nbad=',BAD)
    return GOOD,BAD


def true_rate(tree):
    '''
    输入决策树{}，计算预测的正确率
    '''
    global GOOD,BAD
    GOOD,BAD =[],[]
    count = 0
    good_predict_list,bad_predict_list = find_list1(tree)
    for item in good_predict_list:
        if data[int(item)][-1] == '好瓜': count += 1
    for item in bad_predict_list:
        if data[int(item)][-1] == '坏瓜': count += 1
    if (len(bad_predict_list)+len(good_predict_list)) != 0:
        true_rate = count*100/(len(bad_predict_list)+len(good_predict_list))
        #print(true_rate)
        return true_rate
    else:return 0


def tree_clear(tree):
    '''
    输入决策树,清除测试集
    '''
    for key,value in tree.items():
        if key == '好瓜' or key == '坏瓜':
            value.clear()
            return tree
        else: value = tree_clear(value)
    return tree


def test(tree,test_data):
    '''
    输入决策树，测试集
    返回填充有测试集的决策树
    tree:输入空白/带数据集的决策树
    test_data:测试集
    '''
    for i in range(0,len(test_data)):
        for key,value in tree.items():
            if key == '好瓜' or key == '坏瓜':
                value.append(test_data[i][0])
                if i+1 == len(test_data):return tree
            if key in test_data[i]:
                value = test(value,[test_data[i]])
                if i+1 == len(test_data):return tree


def postpruning(tree):
    '''
    后剪枝
    输入未剪枝的树
    输出后剪枝后的决策树
    '''
    global GOOD2,BAD2
    
    temp = {}
    tree_post = copy.deepcopy(tree)
    for key0,value0 in tree_post.items():
        GOOD2,BAD2 = [],[]
        cycle = True
        if isinstance(value0,dict):
            for key1,value1 in value0.items():
                if isinstance(value1,dict):
                    for key2,value2 in value1.items():
                        if key2 == '好瓜': GOOD2.extend(value2)
                        if key2 == '坏瓜': BAD2 .extend(value2)
                        elif isinstance(value2,dict):
                            tree_post[str(key0)] = postpruning(value0)
                            break
                elif key1 == '好瓜': GOOD2.extend(value1)
                elif key1 == '坏瓜': BAD2 .extend(value1)
            if len(GOOD2)>len(BAD2): key_temp='好瓜'
            if len(GOOD2)<=len(BAD2): key_temp='坏瓜'
            temp[str(key0)] = {}
            (temp[str(key0)])[str(key_temp)] = GOOD2+BAD2
    #print('\n302\ntree_post=',tree_post,'\n\ntemp=',temp)
    if true_rate(tree_post) >= true_rate(temp):return tree_post
    elif true_rate(tree_post) < true_rate(temp):return temp


#-------------------main----------------------------------------------------------------
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

#构建属性值字典
global feature_dict
feature_dict = {}
for a in range(1,7):
    feature_dict[str(a)] = []
    for item in data[1:len(data)]:
        if item[a] not in feature_dict[str(a)]:
            feature_dict[str(a)].append(item[a])

#输入训练集
d = copy.deepcopy(data)
D_train_temp= [d[1]]+[d[2]]+[d[3]]+[d[6]]+[d[7]]+[d[10]]+[d[14]]+[d[15]]+[d[16]]+[d[17]]
D_train = copy.deepcopy(D_train_temp)
D_train_1 = copy.deepcopy(D_train_temp)
tree = treegenerate(D_train,[i for i in range(1,7)])
print('-------Treegenrrate Result--------','\ntree=',tree,'\n------------END-------------')

#输入测试集
GOOD,BAD = [],[]
tree_clear = tree_clear(tree)
D_test_temp = [d[11]]+[d[4]]+[d[5]]+[d[8]]+[d[9]]+[d[12]]+[d[13]]       
D_test = copy.deepcopy(D_test_temp)
test_tree = test(tree_clear,D_test)

#后剪枝
GOOD2,BAD2 = [],[]
tree_postpruning = postpruning(test_tree)
print('\ntest_tree = ',test_tree,'\n\n为剪枝前测试集精度为 ',true_rate(test_tree),'%')
print('\ntree_posted = ',tree_postpruning,'\n\n剪枝后测试集精度为 ',true_rate(tree_postpruning),'%')
