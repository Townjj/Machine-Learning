'''
chapter 9.6
page 214
hierarchical clustering 
AGglomerative NESting (Bottom-up aggregation)
'''

import random
import numpy as np
import matplotlib.pyplot as plt

def load_data40():
    '''
    load watermelon data 4.0
    page202 , table9.1
    '''
    data40=['number','density','sugercontent',
    1,0.697,0.460,
    2,0.774,0.376,
    3, 0.634,0.264,
    4,0.608,0.318,
    5,0.556,0.215,
    6,0.403,0.237,
    7,0.481,0.149,
    8,0.437,0.211,
    9,0.666,0.091,
    10,0.243,0.267,
    11,0.245,0.057,
    12,0.343,0.099,
    13,0.639,0.161,
    14,0.657,0.198,
    15,0.360,0.370,
    16,0.593,0.042,
    17,0.719,0.103,
    18,0.359,0.188,
    19,0.339,0.241,
    20,0.282,0.257,
    21,0.748,0.232,
    22,0.714,0.346,
    23,0.483,0.312,
    24,0.478,0.437,
    25,0.525,0.369,
    26,0.751,0.489,
    27,0.532,0.472,
    28,0.473,0.376,
    29,0.725,0.445,
    30,0.446,0.459]
    data = np.array(data40,dtype=np.dtype).reshape(31,3)
    return data


def hausdorff_distance_min(data,c1,c2):
    '''
    compute the hausdorff distance between two clusters
    and return the d_min
    c1  | cluster c1, including index and data of sample x
    c2  | cluster c2
    '''
    temp2 =[]
    for z in c2:
        temp1 = []
        for x in c1:
            temp1.append(np.sum([(data[x,j]-data[z,j])**2 for j in range(1,3)])**0.5)
        temp2.append(min(temp1))
    dist_h1 = max(temp2)

    temp2 =[]
    for x in c1:
        temp1 = []
        for z in c2:
            temp1.append(np.sum([(data[z,j]-data[x,j])**2 for j in range(1,3)])**0.5)
        temp2.append(min(temp1))
    dist_h2 = max(temp2)
    dist_H = max(dist_h1,dist_h2)
    return dist_H

def dist(x,z):
    return np.sum([(data[z,j]-data[x,j])**2 for j in range(1,3)])**0.5

def min_dist(c1,c2):
    return min(dist(i,j) for i in c1 for j in c2)

def max_dist(c1,c2):
    return max(dist(i,j) for i in c1 for j in c2)

def AGNES_main(data,k):
    '''
    main function of AGNES
    data        | data include head and no.
    k           | num of target clusters
    distance_arr| array of distance above each two clusters
    '''
    m = len(data)-1
    Clusters = [[data[i,0]] for i in range(1,m+1)]
    distance_arr = np.eye(m)

    for i in range(m):
        for j in range(m):
            if i != j:
                distance_arr[i,j] = max_dist(Clusters[i],Clusters[j])
                distance_arr[j,i] = distance_arr[i,j]
    #find the nearest diatance between two clusters and conbine & delete origin clusters       
    q = len(Clusters)
    while q > k :
        min_index = np.where(distance_arr == np.min( distance_arr))
        min_indexi,min_indexj = min_index[0][0],min_index[1][0]
        Clusters[min_indexi] += Clusters[min_indexj]
        del Clusters[min_indexj]
        distance_arr = np.delete(distance_arr,min_indexj,axis=0)
        distance_arr = np.delete(distance_arr,min_indexj,axis=1)
        for j in range(q-1):
            if min_indexi != j:
                distance_arr[min_indexi,j] = max_dist(Clusters[min_indexi],Clusters[j])
                distance_arr[j,min_indexi] = distance_arr[min_indexi,j]
        q -= 1
    return Clusters


def output(data,Clusters):
    '''
    to print & drawing result of DBSCAN
    drawing_data | above whole data in each clusters without head
    Clusters     | result of AGNES only have no. 
    '''
    #print
    k = len(Clusters)
    print('AGNES result:')
    for i in range(k):
        print('>>>Cluster %d including'%i,Clusters[i])

    #drawing 
    drawing_data = []
    for i in range(k):
        drawing_data.insert(i ,np.array([data[j,1:3] for j in Clusters[i]],dtype=np.dtype))
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title('AGNES (d_max  k=5)')
    dot0 = ax1.scatter(drawing_data[0][:,0],drawing_data[0][:,1],c='chocolate',s=2000,alpha=0.8)
    dot1 = ax1.scatter(drawing_data[1][:,0],drawing_data[1][:,1],c='c',s=2000,alpha=0.8)
    dot2 = ax1.scatter(drawing_data[2][:,0],drawing_data[2][:,1],c='blueviolet',s=2000,alpha=0.8)
    dot3 = ax1.scatter(drawing_data[3][:,0],drawing_data[3][:,1],c='gold',s=2000,alpha=0.8)
    dot4 = ax1.scatter(drawing_data[4][:,0],drawing_data[4][:,1],c='dimgray',s=2000,alpha=0.8)
    plt.xlabel('density')
    plt.ylabel('sugercontent')
    plt.show()


data = load_data40()
Clusters = AGNES_main(data,k=5)
output(data,Clusters)
