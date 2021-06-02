# Chapter 9 Clustering
## 9.1 聚类任务
聚类(clustering)是一种研究多、应用广的一种无监督学习(unsupervised learning)算法。将数据集中的样本划分为若干不相交子集，每个子集亦成为‘簇’(cluster)。  
样本集含有m个n维的无标记样本：
$$D=\left\{ \boldsymbol{x}_1,\boldsymbol{x}_{2,...,}\boldsymbol{x}_m \right\}$$
其中 $\boldsymbol{x}_i=\left( x_{i1};x_{i2};...;x_{in} \right)$为n维向量  
聚类算法运行后将样本集划分为k个不相交的簇：
$$\left\{ C_l\,\, \mid \,\,l=1,2,...,k \right\}$$
则样本$\boldsymbol{x}_j$的簇标记(cluster label)为:
$$\lambda _j\in \left\{ 1,2,...,k \right\}$$
将每个样本的簇标记整合成向量，得到样本集的聚类结果：
$$\boldsymbol{\lambda }=\left( \lambda _1,\lambda _2,...,\lambda _m \right)$$    

## 9.2 性能度量
聚类性能度量亦称聚类有效性指标(validty index)，用以评价聚类结果的好坏，有些亦可作为聚类过程的优化目标。  
+ 外部指标(external index)  
将聚类结果与参考模型(reference model)比较，当聚类划分的簇为$\boldsymbol{C}=\left\{ C_1,C_2,...,C_k \right\}$，参考模型划分的簇为$\boldsymbol{C}^*=\left\{ C_{1}^{*},C_{2}^{*},...,C_{k}^{*} \right\}$时，定义如下指标：  
在C中隶属**相同**簇，在C\*中隶属**相同**簇的样本对集合SS（Same Same）：  $a=\left| SS \right|, \left| SS \right|=\left\{ \left( x_j,x_j \right) \,\, \mid \,\,\lambda _i=\lambda _j,\lambda _{i}^{*}=\lambda _{j}^{*},i<j \right\}$  
在C中隶属**相同**簇，在C\*中隶属**不同**簇的样本对集合SD（Same Different）： $b=\left| SS \right|, \left| SS \right|=\left\{ \left( x_j,x_j \right) \,\, \mid \,\,\lambda _i=\lambda _j,\lambda _{i}^{*}\ne \lambda _{j}^{*},i<j \right\}$  
在C中隶属**不同**簇，在C\*中隶属**相同**簇的样本对集合DS（Different Same）： $c=\left| SS \right|, \left| SS \right|=\left\{ \left( x_j,x_j \right) \,\, \mid \,\,\lambda _i\ne \lambda _j,\lambda _{i}^{*}=\lambda _{j}^{*},i<j \right\}$  
在C中隶属**不同**簇，在C\*中隶属**不同**簇的样本对集合DD（Different Different）：$d=\left| SS \right|, \left| SS \right|=\left\{ \left( x_j,x_j \right) \,\, \mid \,\,\lambda _i\ne \lambda _j,\lambda _{i}^{*}\ne \lambda _{j}^{*},i<j \right\}$








