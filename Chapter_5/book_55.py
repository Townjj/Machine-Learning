'''
实现标准BP与累积BP算法训练单隐层网络并比较
2021/05/06-2021/05/07
'''

import numpy as np
import random
import matplotlib.pyplot as plt

def sigmoid(a):
        return 1/(1+np.exp(-a))


def standard(x_train,y_train,iteration=200,eta=0.1):
        n,d = x_train.shape  #num of x_test dataset / dim of x_test dataset
        q = d+2  #the num of hidden layer
        v = np.random.random((d,q))
        theta_1 = np.random.random((1,q))
        w = np.random.random((1,q))
        theta_2 = np.random.random((1,1))
        cost_list = []
        while iteration > 0:
                for i in range(0,n):
                        error_k = 0
                        b = sigmoid(x_train[i,:].reshape(1,d)@v-theta_1)
                        y_predict = sigmoid(w@b.T-theta_2)
                        #print('y_predict= ',y_predict)
                        error_k = ((y_predict-y_train[i])**2)/2
                        g = y_predict*(1-y_predict)*(y_train[i]-y_predict) #gradient decent
                        w += eta*g*b
                        theta_2 += (-eta*g)
                        eh = b*(1-b)*(w*g)
                        v += eta*(x_train[i,:].reshape(d,1))@eh
                        theta_1 += (-eta*eh)
                cost_list.append(error_k[0,0])
                print ('error_k = ',error_k)
                iteration -= 1
        return v,theta_1,w,theta_2,cost_list


def accumulated(x_train,y_train,iteration=200,eta=0.1):
        n,d = x_train.shape  #num of x_test dataset / dim of x_test dataset
        q = d+2  #the num of hidden layer elements
        v = np.random.random((d,q))
        theta_1 = np.random.random((1,q))
        w = np.random.random((1,q))
        theta_2 = np.random.random((1,1))
        cost_list = []
        while iteration > 0:
                w_des,theta_2_des,v_des,theta_1_des =0,0,0,0
                for i in range(0,n):
                        error = 0
                        b = sigmoid(x_train[i,:].reshape(1,d)@v-theta_1)
                        y_predict = sigmoid(w@b.T-theta_2)
                        #print('y_predict= ',y_predict)
                        error = ((y_predict-y_train[i])**2)/2
                        g = y_predict*(1-y_predict)*(y_train[i]-y_predict) #gradient decent
                        w_des += eta*g*b
                        theta_2_des += (-eta*g)
                        eh = b*(1-b)*(w*g)
                        v_des += eta*(x_train[i,:].reshape(d,1))@eh
                        theta_1_des += (-eta*eh)
                w += w_des
                theta_2 += theta_2_des
                v += v_des
                theta_1 += theta_1_des
                iteration -= 1
                cost_list.append(error[0,0])
                print('error = ',error)
        return v,theta_1,w,theta_2,cost_list

def test(x_test,y_test,v,theta_1,w,theta_2):
        n,d = x_test.shape
        true_num = 0
        for i in range(0,n):
                b = sigmoid(x_test[i,:].reshape(1,d)@v-theta_1)
                y_predict = sigmoid(w@b.T-theta_2)
                if y_predict > 0.5 and y_test[i] == 1:
                        true_num += 1
                if y_predict <= 0.5 and y_test[i] ==0:
                        true_num += 1
                print(y_predict,y_test[i])
        print('\ntrue rate = ',true_num*100/len(y_test),'%')
        return true_num/len(y_test)


data_x=[[2, 3, 3, 2, 1, 2, 3, 3, 3, 2, 1, 1, 2, 1, 3, 1, 2],
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 1, 2, 2, 2, 1, 1],
        [2, 3, 2, 3, 2, 2, 2, 2, 3, 1, 1, 2, 2, 3, 2, 2, 3],
        [3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 1, 1, 2, 2, 3, 1, 2],
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 1, 1, 2, 3, 2],
        [1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1],
        [0.697, 0.774, 0.634, 0.668, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360,0.593, 0.719],
        [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370,0.042, 0.103]]
data_x = np.array(data_x).T  
data_y = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
iteration = 500
learning_rate = 0.01
v_acc, theta_1_acc, w_acc, theta_2_acc, cost_list_acc =accumulated(data_x[5:15,:],data_y[5:15],iteration=iteration,eta=learning_rate)
v_sta, theta_1_sta, w_sta, theta_2_sta, cost_list_sta =standard(data_x[5:15,:],data_y[5:15],iteration=iteration,eta=learning_rate)
test(np.vstack((data_x[0:5,:],data_x[15:17,:])),np.hstack((data_y[0:5],data_y[15:17])),v_acc, theta_1_acc, w_acc, theta_2_acc)
test(np.vstack((data_x[0:5,:],data_x[15:17,:])),np.hstack((data_y[0:5],data_y[15:17])),v_sta, theta_1_sta, w_sta, theta_2_sta)

plt.figure(figsize=(10,12),dpi=100)
plt.plot(range(0,iteration),cost_list_acc,label='Accumulated BP',color='g')
plt.plot(range(0,iteration),cost_list_sta,label='Standard BP',color='r')
plt.legend()
plt.title("Neural Network (Learning rate=0.01)")
plt.xlabel("iteration")
plt.ylabel("cost")
plt.show()
