'''
practice 9.4 in <machine learning>
k-means algorithm
'''

import numpy as np
import random
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


def distance(a,b):
    '''
    Calculate vector (sample) distance 
    a,b   | two input vectors 
    '''
    return np.sum([(a[i]-b[i])**2 for i in range(len(a))])**0.5


def k_means(data,k=3,iterations=200,m=30):
    '''
    k-means algorithm
    data        | Training set (including header) 
    k           | Number of clusters 
    iterations  | Training times 
    m           | Number of training samples 
    '''
    mu_list = np.array([data[i,1:] for i in random.sample(range(1,30),k)])
    
    while iterations:

        #Clustering based on distance between each sample and cluster center vectors
        Clusters = [[],[],[]]
        for i in range(1,m+1): 
            distance_list = [ distance(data[i,1:],mu_list[j]) for j in range(k) ]
            lambda_i = np.argmin(distance_list)
            Clusters[lambda_i].append(data[i,0])

        #update each cluster center vectors based on sample in each cluster
        Clusters_data = [[],[],[]]
        for i in range(k):
            Clusters_data[i] = [data[j,1:] for j in Clusters[i]]
            mu_list[i] = np.mean(Clusters_data[i],axis=0)
        iterations -= 1
    return Clusters,Clusters_data


def output(drawing_data,k=3):
    '''
    Data visualization 
    drawing_data | Sample set (without header)
    k            | Number of clusters
    '''
    print('k means result:')
    for i in range(k):
        print('>>>Cluster %d including'%i,result[i],)

    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title('k-means (iteration=200,k=3)')
    for i in range(k):
        drawing_data[i] = np.array(drawing_data[i],dtype=np.dtype)
    ax1.scatter(drawing_data[0][:,0],drawing_data[0][:,1],c='r',s=150)
    ax1.scatter(drawing_data[1][:,0],drawing_data[1][:,1],c='g',s=150)
    ax1.scatter(drawing_data[2][:,0],drawing_data[2][:,1],c='b',s=150)
    plt.xlabel('density')
    plt.ylabel('sugercontent')
    plt.show()

## Main func
data = load_data40()
k,iterations,m=3,200,30
result,drawing_data = k_means(data)
drawing_data = np.array(drawing_data,dtype=np.dtype)
output(drawing_data=drawing_data)