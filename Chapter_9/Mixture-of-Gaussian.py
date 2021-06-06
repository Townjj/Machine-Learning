'''
chapter 9.4.3
page 206
hierarchical clustering 
Gaussian mixture clustering
'''

import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

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


def p(x,mu,mat):
    return np.exp(-0.5*(x-mu).T@(np.linalg.inv(mat))@(x-mu))/(2*np.pi*(np.linalg.det(mat)**0.5))

def pm(data,alpha,mu,con_arr,i,j,k):
    #print(alpha[i],data[j,1:],mu[i],con_arr[i,:,:],con_arr[:,:,:])
    a = alpha[i]*p(data[j,1:],mu[i],con_arr[i,:,:])
    b = np.sum([alpha[l]*p(data[j,1:],mu[l],con_arr[l,:,:]) for l in range(k)])
    #print('a=',a,'b=',b)
    return a/b
    

def Gaussian_clustering(data,k,iteration=3):
    '''
    main function of Gaussian clustering algorithm
    data |
    k    |
    con_arr  | 协方差矩阵
    '''
    m = len(data)-1 #m=30
    alpha = [(1/3)]* 3
    mu = [data[6,1:],data[22,1:],data[27,1:]]
    con_arr = np.array([[[0.1,0],[0,0.1]],[[0.1,0],[0,0.1]],[[0.1,0],[0,0.1]]])
    y_arr = np.zeros((m+1,k))
    print(pm(data,alpha,mu,con_arr,i=0,j=1,k=3))
    ###pm is right 
    ###maybe line 86-92 is wrong
    '''
    while iteration:
        for j in range(1,m+1):
            #print('j=',j)
            for i in range(k):
                y_arr[j,i] = pm(data,alpha,mu,con_arr,i,j,k)
        print(y_arr)
        for i in range(k):
            temp = np.sum([y_arr[j,i] for j in range(1,m+1)])
            for j in range(1,m+1):
                mu[i] += y_arr[j,i]*(np.array(data[j,1:])) / temp
            for j in range(1,m+1):
                con_arr[i,:,:] += y_arr[j,i]*((data[j,1:]-mu[i])@(data[j,1:]-mu[i]).T) /temp
            alpha[i] = temp/m
        #print('iteration = ',iteration,'alpha=',alpha,'\nmu=',mu,'\ncon_arr=',con_arr)
        iteration -= 1
    Clusters = [[],[],[]]
    for j in range(1,m+1):
        label = np.argmax([y_arr[j,i] for i in range(k)])
        Clusters[label].append(j)
        print('89',Clusters)
    return Clusters
    '''
    



data = load_data40()
k = 3
Gaussian_clustering(data,k=k)
