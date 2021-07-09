import numpy as np
import matplotlib.pyplot as plt
import math

def Unit_vector(a):
    '求a单位向量'
    unit_vector = a/np.linalg.norm(a)
    return unit_vector

def projection(a,w):
    '''
    矩阵a投影到直线w中，返回投影后的矩阵b
    n = x集合数、w个数
    m = x个数
    w = (1,n)
    a = (m,n)
    b = (m,n)
    '''
    
    w_uni = Unit_vector(w) #1,n
    temp1 = w_uni@a.T    #1,n x n,m = 1,m
    b = w_uni.T@temp1    #n,1 x 1,m = n,m
    return b.T            

water_melon_data = np.array([[0.697,0.460,1],
                              [0.774,0.376,1],
                              [0.634,0.264,1],
                              [0.608,0.318,1],
                              [0.556,0.215,1],
                              [0.403,0.237,1],
                              [0.481,0.149,1],
                              [0.437,0.211,1],
                              [0.666,0.091,0],
                              [0.243,0.267,0],
                              [0.245,0.057,0],
                              [0.343,0.099,0],
                              [0.639,0.161,0],
                              [0.657,0.198,0],
                              [0.360,0.370,0],
                              [0.593,0.042,0],
                              [0.719,0.103,0]])

#数据处理
data_X = water_melon_data[:,0:2]
data_y = water_melon_data[:,2]
data_x_true = np.ones((8,2))
data_x_false = np.ones((9,2))
j,k = 0,0
for i in range(0,17):
    if data_y[i] == 1:
        data_x_true[j,:] = data_X[i,:]
        j+=1
    else:
        data_x_false[k,:] = data_X[i,:]
        k+=1


u0 = np.sum(data_x_false,axis=0)/(data_x_false.shape[0])
u1 = np.sum(data_x_true,axis=0)/(data_x_true.shape[0])
S_w=np.zeros((2,2))
for i in range(0,17):
    x_temp = data_X[i].reshape(2,1)
    if data_y[i] == 0:u_temp = u0.reshape(2,1)
    if data_y[i] == 1:u_temp = u1.reshape(2,1)
    S_w += np.dot(x_temp-u_temp,(x_temp-u_temp).T)
cov1 = np.cov(data_x_true.T)
cov0 = np.cov(data_x_false.T)
#S_w = (cov0+cov1)   #2，2
print('\ncov0=',cov0,'\ncov1=',cov1,'\nS_w=',S_w)
u0_u1 = (u0-u1).reshape(2,1)
Sigma_mat = np.zeros((S_w.shape[0],S_w.shape[1]))
U,Sigma,V_T = np.linalg.svd(S_w)
Sigma_mat = np.diag(Sigma)  #2,2
S_w_1 = V_T.T@np.linalg.inv(Sigma_mat)@U.T #2,2
print('S_w_1=',S_w_1) 
w = S_w_1@(u0_u1)      #2,2x2,1= 2,1
S_b = (u0_u1)@((u0_u1).T)  #2,1x1,2=2,2
print('\nu0=',u0,'\nu1=',u1,'\nw=',w,'\nu0-u1=',u0_u1)
print(w.shape,'\nsb=',S_b,'\n\nsw=',S_w)
a = w.T@S_b@w  #1,2x2,2x2,1=1,1
b = w.T@S_w@w   #1,2x2,2x2,1=1,1
J = a/b
print('J=',J)

#绘图
plt.xlim( 0, 1 )
plt.ylim( 0, 0.7 )
plt.scatter(data_x_true[:,0],data_x_true[:,1],c='red',marker='s')
plt.scatter(data_x_false[:,0],data_x_false[:,1],c='green',marker='*')
x_sample1 = np.arange(0,1.0,0.1)
x_sample2 = (w[1]*x_sample1)/w[0]
plt.plot(x_sample1,x_sample2,'black') 
plt.show()
plt.close('all')