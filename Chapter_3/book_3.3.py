import numpy as np
import matplotlib.pyplot as plt
import math


def likelihood(x,y,beita):
    i=0
    l=0
    for i in range(1,17):
        l=l-y[i,0]*(x[i,:]@beita.T)+np.log(1+np.exp(x[i,:]@beita.T))
        i+=1
    return l

water_melon_data = np.matrix([[0.697,0.460,1],
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

data_X=water_melon_data[:,0:2]
data_y=water_melon_data[:,2]
X=np.hstack((data_X,np.ones(shape=(17,1),dtype=float))) #17x3
w=np.random.rand(2,1)
b=np.random.rand(1,1)
beita=np.matrix(np.vstack((w,b))) #3x1

#筛选正反类
plot_truex1=[]
plot_truex2=[]
plot_falsex1=[]
plot_falsex2=[]
for i in range(0,17):
    if data_y[i,0] == 1:
        plot_truex1.append(data_X[i,0])
        plot_truex2.append(data_X[i,1]) 
    else:
        plot_falsex1.append(data_X[i,0])
        plot_falsex2.append(data_X[i,1])

#迭代beita
beita_t=beita.T
for j in range(0,50):
    p1=np.exp(X@beita_t.T)/(1+np.exp(X@beita_t.T)) #17x3*3*1=17*1
    #beita 一阶导数
    beita_f1=np.zeros((1,3))
    for i in range(0,17):
        beita_f1=beita_f1-X[i,:]*(data_y[i,0]-p1[i,0])
    #beita 二阶导数
    xx=(X@X.T)
    beita_f2=0
    for i in range(0,17):
        beita_f2=beita_f2+xx[i,i]*p1[i,0]*(1-p1[i,0])
    #更新beita
    beita_t=beita_t-(beita_f2)**(-1)*(beita_f1)
    #print('\n--------------','j=',j,'--------------')
    #print('beita_t=',beita_t)
    print('likelihood=',likelihood(X,data_y,beita_t))

plt.scatter(plot_truex1,plot_truex2,c='red',marker='o')
plt.scatter(plot_falsex1,plot_falsex2,c='green',marker='o')
x_sample1=np.arange(0,1.0,0.1)
y_sample=-(beita_t[0,0]*x_sample1+beita_t[0,2])/beita_t[0,1]
plt.plot(x_sample1,y_sample,'black') 
    
plt.show()