import numpy as np
import matplotlib.pyplot as plt


def linear_kernel(a,b):
    return a.T @ b

def clip(a,L,H):
    if a > H: return H
    if L <= a <= H: return a
    else: return L

data = [[0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.403, 0.237, 1],
        [0.481, 0.149, 1],
        [0.437, 0.211, 1],
        [0.666, 0.091, -1],
        [0.243, 0.267, -1],
        [0.245, 0.057, -1],
        [0.343, 0.099, -1],
        [0.639, 0.161, -1],
        [0.657, 0.198, -1],
        [0.360, 0.370, -1],
        [0.593, 0.042, -1],
        [0.719, 0.103, -1]]
data = np.array(data)
data_x = data[:,0:2]
data_y = data[:,-1]
train_x = data_x[:]  #17*2
train_y = data_y[:]
n , d = train_x.shape[0],train_x.shape[1]

w = np.array(np.random.rand(2,1))
b = np.array(np.random.rand(n))
alpha = np.array(np.random.rand(n))
y_pre = np.array(np.random.rand(n))
iter = 200
while iter > 0:
    for i in range(1,n):
        y_pre = np.array(np.zeros(n))
        for i in range(n): 
            y_pre += alpha[i] * train_y[i] * linear_kernel(train_x[i],train_x.T)
        y_pre += b
        print (y_pre)
        error_y = y_pre - train_y
        alpha_y = alpha * train_y

        alpha_old = alpha[:]
        eta = linear_kernel(train_x[i-1],train_x[i-1]) + linear_kernel(train_x[i],train_x[i]) + 2*linear_kernel(train_x[i-1],train_x[i])
        alpha_i_unclip = alpha[i] + train_y[i]*(error_y[i-1]-error_y[i])/eta 
        C = -np.sum(alpha_y) + alpha_y[i-1] + alpha_y[i]
        #update alpha
        if train_y[i-1] != train_y[i]: 
            alpha_y[i] = clip(alpha_i_unclip, max(0,alpha[i]-alpha[i-1]), min(C,C+alpha[i]-alpha[i-1]))
        if train_y[i-1] == train_y[i]:
            alpha_y[i] = clip(alpha_i_unclip, max(0,alpha[i]-alpha[i-1]-C), min(C,alpha[i]+alpha[i-1]))
        alpha[i-1] = alpha[i-1] + train_y[i-1]*train_y[i]*(alpha_i_unclip-alpha[i])
        #update b[i-1] & b[i]
        if 0 < alpha[i-1] < C: 
            b[i-1] = -error_y[i-1] + train_y[i-1]*linear_kernel(train_x[i-1],train_x[i-1])*(alpha_old[i-1]-alpha[i-1]) \
            - train_y[i]*linear_kernel(train_x[i],train_x[i-1])*(alpha_old[i]-alpha[i]) + b[i-1]
        if 0 < alpha[i] < C: 
            b[i] = -error_y[i] + train_y[i-1]*linear_kernel(train_x[i-1],train_x[i])*(alpha[i-1]-alpha_old[i-1]) \
            - train_y[i]*linear_kernel(train_x[i],train_x[i])*(alpha[i]-alpha_old[i]) + b[i]
        elif (alpha[i-1] == 0 or alpha[i-1] == C) and (alpha[i] == 0 and alpha[i] == C):
            b[i-1] = b[i] = (b[i-1]+b[i])/2
    print(alpha,b)
    iter -=1
#w=求和(ai*yi*xi),求和对象是支持向量，即，ai>0的样本点，xi，yi为支持向量对应的label和data"""
w=np.mat(np.zeros((1,d)))
for i in range(n):
    if alpha[i]>0:
        w+=train_y[i]*alpha[i]*train_x[i,:]
print('\nw=',w)

#drawing
plt.scatter(train_x[:8,0], train_x[:8, 1], label='1')
plt.scatter(train_x[8:18,0], train_x[8:18, 1], label='-1')
x = np.array(np.arange(0,1,0.1))
y = (-np.mean(b)-w[0,0]*x)/w[0,1] #由w1*x1+w2*x2+b=0得到x2(即y)=(-b-w1x1)/w2
plt.plot(x,y,color='g',linewidth=3.0,label="Boarder") 
plt.xlabel('Midu')
plt.ylabel('Hantangliang')
plt.title('SMO')
plt.legend()
plt.show()
