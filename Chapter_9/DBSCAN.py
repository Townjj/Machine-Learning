'''
density based spatical clustering of applications with noise
chapter 9.5
pages   211
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

def compute_Pts(data,x,epsilon):
    '''
    compute the number of samples in x's epsilon neigh-borhood
    data    | the sample set(with head) need to test if in x's epsilon neigh-borhood (3,31)
    x       | core object (1,3) with no.*
    epsilon | diatance between x and neigh sample
    '''
    num = 0
    target = []
    for i in range(1,len(data)):
        if np.sum([(data[i,j]-x[j])**2 for j in range(1,len(x))])**0.5 <= epsilon:
            num += 1
            target.append(data[i,:])
    return [num,np.array(target)]

def DBSCAN_main(data,epsilon,MinPts):
    '''
    main function to clustering
    data | the data need to cluster
    core_object | index of core_object unvisit
    unvisit     | index of unvisit sample
    queue       | temp list to padding the neigh sample in queue[index] epsilon area into C[k]

    '''
    core_object = list([data[core_index,0] for core_index in range(1,31) if compute_Pts(data,data[core_index,:],epsilon)[0] >= MinPts ])
    k = 0
    unvisit = list(data[1:,0])
    Clusters = []
    # find out the each density-connect area (cluster) in each core_object
    while core_object:
        unvisit_temp = unvisit.copy()
        queue = list(data[random.sample(core_object,1),0])
        unvisit.remove(queue[0])

        ## padding the neigh sample in queue[index] epsilon area into C[k]
        index = 0
        while index < len(queue):      #cation:len(queue) may change after each iteration 
            [neigh_num,neigh_data] = compute_Pts(data,data[queue[index],:],epsilon)
            if  neigh_num >= MinPts:
                temp1 = list(set(list(neigh_data[:,0])) & set(unvisit) )
                queue += temp1
                unvisit = list(set(unvisit) - set(temp1))
            index += 1

        ## padding cluster and delete them in core_object
        Clusters.insert(k, list(set(unvisit_temp) - set(unvisit)))
        core_object = list(set(core_object) - set(Clusters[k]))
        k += 1
    return Clusters


def output(data,Clusters):
    '''
    to print & drawing result of DBSCAN
    drawing_data | above whole data in each clusters without head
    problem : k is uncertain(but always k=4), so the code 121-125 to plot is need to improve
    '''
    #print
    k = len(Clusters)
    print('DBSCAn result:')
    for i in range(k):
        print('>>>Cluster %d including'%i,Clusters[i])

    #drawing 
    drawing_data = []
    for i in range(k):
        drawing_data.insert(i ,np.array([data[j,1:3] for j in Clusters[i]],dtype=np.dtype))
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title('DBSCAN (epsilon=0.10 MinPts=5)')
    dot0 = ax1.scatter(drawing_data[0][:,0],drawing_data[0][:,1],c='chocolate',s=2000,alpha=0.8)
    dot1 = ax1.scatter(drawing_data[1][:,0],drawing_data[1][:,1],c='c',s=2000,alpha=0.8)
    dot2 = ax1.scatter(drawing_data[2][:,0],drawing_data[2][:,1],c='blueviolet',s=2000,alpha=0.8)
    dot3 = ax1.scatter(drawing_data[3][:,0],drawing_data[3][:,1],c='gold',s=2000,alpha=0.8)
    #dot4 = ax1.scatter(drawing_data[4][:,0],drawing_data[4][:,1],c='dimgray',s=150)
    plt.xlabel('density')
    plt.ylabel('sugercontent')
    plt.show()

#Main func
data = load_data40()
Clusters_result = DBSCAN_main(data,epsilon=0.10,MinPts=5)
output(data,Clusters_result)




        


