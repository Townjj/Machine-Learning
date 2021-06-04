'''
Learning Vector Quantization algorithm in <machine learning>
'''

import numpy as np
import random
import matplotlib.pyplot as plt

def load_data40():
    '''
    load watermelon data 4.0 (with labels)
    page202 , table9.1
    '''
    data40=['number','density','sugercontent','labels',
    1,0.697,0.460,1,
    2,0.774,0.376,1,
    3, 0.634,0.264,1,
    4,0.608,0.318,1,
    5,0.556,0.215,1,
    6,0.403,0.237,1,
    7,0.481,0.149,1,
    8,0.437,0.211,1,
    9,0.666,0.091,0,
    10,0.243,0.267,0,
    11,0.245,0.057,0,
    12,0.343,0.099,0,
    13,0.639,0.161,0,
    14,0.657,0.198,0,
    15,0.360,0.370,0,
    16,0.593,0.042,0,
    17,0.719,0.103,0,
    18,0.359,0.188,0,
    19,0.339,0.241,0,
    20,0.282,0.257,0,
    21,0.748,0.232,0,
    22,0.714,0.346,0,
    23,0.483,0.312,1,
    24,0.478,0.437,1,
    25,0.525,0.369,1,
    26,0.751,0.489,1,
    27,0.532,0.472,1,
    28,0.473,0.376,1,
    29,0.725,0.445,1,
    30,0.446,0.459,1]
    data = np.array(data40,dtype=np.dtype).reshape(31,4)
    return data


def distance(a,b):
    '''
    Calculate vector (sample) distance 
    a,b   | two input vectors 
    '''
    return np.sum([(a[i]-b[i])**2 for i in range(len(a))])**0.5


def LVQ(data,learning_rate=0.01,iteration=1000,q=5):
    '''
    Learning Vector Quantization function
    data            | Training set (including header) 
    q               | Number of clusters 
    iterations      | Training times 
    learning _rate  | Number of clusters 

    '''
    #update prototype vector
    prototype_vector = np.array([data[i,1:] for i in random.sample(range(1,30),q)])
    print(prototype_vector)
    while iteration:
        random_sample_index = random.randint(1,30)
        distance_list = [distance(data[random_sample_index ,1:2],prototype_vector[i,0:2]) for i in range(q)]
        nearest_index = np.argmin(distance_list)
        if data[random_sample_index,-1] == prototype_vector[nearest_index,-1]:
            prototype_vector[nearest_index,0:2] += learning_rate*(data[random_sample_index ,1:3]-prototype_vector[nearest_index,:2])
        else:
            prototype_vector[nearest_index,0:2] -= learning_rate*(data[random_sample_index ,1:3]-prototype_vector[nearest_index,:2])
        iteration -= 1

    #padding Clusters
    Clusters = [[],[],[]]
    for i in range(1,len(data)):
        distance_list = [distance(data[i ,1:2],prototype_vector[j,0:2]) for j in range(q)]
        nearest_index = np.argmin(distance_list)
        Clusters[nearest_index].append(data[i,0])
    #padding Clusters data to drawing
    Clusters_data = [[],[],[]]
    for i in range(q):
        Clusters_data[i] = [data[j,1:3] for j in Clusters[i]]

    return prototype_vector,Clusters,Clusters_data


def output(prototype_vector_result,drawing_data,Clusters,q=5):
    '''
    to print & drawing result of LVQ
    drawing_data | above whole data in each clusters
    '''
    #print
    print('k means result:')
    for i in range(q):
        print('>>>Cluster %d including'%i,Clusters[i])

    #drawing
    drawing_labels = ['Good' if prototype_vector_result[i,-1] == 1 else 'Bad' for i in range(q)]
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title('LVQ (iteration=1000, learning rate = 0.01)')
    for i in range(q):
        drawing_data[i] = np.array(drawing_data[i],dtype=np.dtype)
    dot0 = ax1.scatter(drawing_data[0][:,0],drawing_data[0][:,1],c='r',s=150)
    dot1 = ax1.scatter(drawing_data[1][:,0],drawing_data[1][:,1],c='g',s=150)
    dot2 = ax1.scatter(drawing_data[2][:,0],drawing_data[2][:,1],c='b',s=150)

    plt.legend(handles=[dot0,dot1,dot2],labels=drawing_labels)
    plt.xlabel('density')
    plt.ylabel('sugercontent')
    plt.show()


#main function
data = load_data40()
prototype_vector_result,Clusters_result ,Clusters_data_result= LVQ(data)
drawing_data = np.array(Clusters_data_result,dtype=np.dtype)
output(prototype_vector_result, Clusters_data_result, Clusters_result)

    