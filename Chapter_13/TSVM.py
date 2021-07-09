'''
半监督SVM
'''
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import svm 

def load_data40():
    '''
    load watermelon data 4.0 (with labels)
    page202 , table9.1
    '''
    data40=['number','density','sugercontent','labels',
    1,0.697,0.460,1,
    2,0.774,0.376,1,
    3,0.634,0.264,1,
    4,0.608,0.318,1,
    5,0.556,0.215,1,
    6,0.403,0.237,1,
    7,0.481,0.149,1,
    8,0.437,0.211,1,
    9,0.666,0.091,-1,
    10,0.243,0.267,-1,
    11,0.245,0.057,-1,
    12,0.343,0.099,-1,
    13,0.639,0.161,-1,
    14,0.657,0.198,-1,
    15,0.360,0.370,-1,
    16,0.593,0.042,-1,
    17,0.719,0.103,-1,
    18,0.359,0.188,-1,
    19,0.339,0.241,-1,
    20,0.282,0.257,-1,
    21,0.748,0.232,-1,
    22,0.714,0.346,-1,
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

#load data
data40 = load_data40()
dataTest = data40[19:27,:]
dataLabel = np.vstack((data40[1:18,:],data40[27:31,:]))
randomList = [round(random.random(),3) for i in range(120)]
dataUnlabel = np.array(randomList).reshape(60,2)
#print(dataTest,dataLabel,dataUnlabel) 

# init SVM
clf = svm.SVC(C=100,kernel='linear')
clf.fit(dataLabel[:,1:3],dataLabel[:,-1].astype('int'))
fakeLabel = clf.predict(dataUnlabel)

# init SVM drawing
unlabeled_positive_x = []
unlabeled_positive_y = []
unlabeled_negative_x = []
unlabeled_negative_y = []
for i in range(len(dataUnlabel)):
    if int(fakeLabel[i]) == 1:
        unlabeled_positive_x.append(dataUnlabel[i, 0])
        unlabeled_positive_y.append(dataUnlabel[i, 1])
    else:
        unlabeled_negative_x.append(dataUnlabel[i, 0])
        unlabeled_negative_y.append(dataUnlabel[i, 1])

# main
Cu = 0.01
Cl = 1
weight = np.ones(len(dataLabel)+len(dataUnlabel))
weight[len(dataUnlabel):] = Cu
print(dataLabel[:,1:3].shape,dataUnlabel.shape)
trainSample = np.concatenate((dataLabel[:,1:3],dataUnlabel),axis=0)
trainLabel = np.concatenate((dataLabel[:,-1].astype('int'),fakeLabel),axis=0)
unlabelId = np.arange(len(dataUnlabel))
iteration = 50
while Cu < Cl:
    clf.fit(trainSample,trainLabel,sample_weight=weight)
    while True:
        iteration -= 1
        predictY = clf.decision_function(dataUnlabel)
        epsilon = 1-predictY*fakeLabel
        positiveSet , positiveId = epsilon[fakeLabel>0] , unlabelId[fakeLabel>0]
        negativeSet , negativeId = epsilon[fakeLabel<0] , unlabelId[fakeLabel<0]
        positiveMaxId = positiveId[np.argmax(positiveSet)]
        negativeMaxId = negativeId[np.argmax(negativeSet)]
        epsilon1 , epsilon2 = epsilon[positiveMaxId] , epsilon[negativeMaxId]
        print(Cu,Cl)
        if epsilon1>0 and epsilon2>0 and epsilon1+epsilon2>=2:
            fakeLabel[positiveMaxId] = -fakeLabel[positiveMaxId]
            fakeLabel[negativeMaxId] = -fakeLabel[negativeMaxId]
            trainLabel = np.hstack((dataLabel[:,-1].astype('int'),fakeLabel))
            clf.fit(trainSample,trainLabel,sample_weight=weight)
        else:
            break
    Cu = min(2*Cu,Cl)
    weight[len(dataUnlabel):] = Cu 
    

# drawing
plt.scatter(unlabeled_positive_x, unlabeled_positive_y, color='red', s=15)
plt.scatter(unlabeled_negative_x, unlabeled_negative_y, color='blue', s=15)
xpoint0 = np.linspace(0,1,10)
ypoint0 = -(clf.coef_[0][0] * xpoint0 + clf.intercept_) / clf.coef_[0][1]
plt.plot(xpoint0, ypoint0, color='green')

xpoint = np.linspace(0,1,10)
ypoint = -(clf.coef_[0][0]*xpoint+clf.intercept_) / clf.coef_[0][1]
plt.plot(xpoint,ypoint,color='green')
plt.show()