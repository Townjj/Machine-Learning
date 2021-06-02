<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$
\\(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\)

# Chapter 9 Clustering
## 9.1 聚类任务
聚类(clustering)是一种研究多、应用广的一种无监督学习(unsupervised learning)算法。将数据集中的样本划分为若干不相交子集，每个子集亦称为‘簇’(cluster)。  
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
将聚类结果与参考模型(reference model)比较，当聚类划分的簇为$\boldsymbol{C}=\left\{ C_1,C_2,...,C_k \right\}$，参考模型划分的簇为$\boldsymbol{C}^*=\left\{ C_{1}^{*},C_{2}^{*},...,C_{k}^{*} \right\}$时，定义：   
在C中隶属**相同**簇，在C\*中隶属**相同**簇的样本对集合SS（Same Same）:  $a=\left| SS \right|, \left| SS \right|=\left\{ \left( x_j,x_j \right) \,\, \mid \,\,\lambda _i=\lambda _j,\lambda _{i}^{*}=\lambda _{j}^{*},i<j \right\}$  
在C中隶属**相同**簇，在C\*中隶属**相同**簇的样本对集合SS（Same Same）：  $a=\left| SS \right|, \left| SS \right|=\left\{ \left( x_j,x_j \right) \,\, \mid \,\,\lambda _i=\lambda _j,\lambda _{i}^{*}=\lambda _{j}^{*},i<j \right\}$  
在C中隶属**相同**簇，在C\*中隶属**不同**簇的样本对集合SD（Same Different）： $b=\left| SS \right|, \left| SS \right|=\left\{ \left( x_j,x_j \right) \,\, \mid \,\,\lambda _i=\lambda _j,\lambda _{i}^{*}\ne \lambda _{j}^{*},i<j \right\}$  
在C中隶属**不同**簇，在C\*中隶属**相同**簇的样本对集合DS（Different Same）： $c=\left| SS \right|, \left| SS \right|=\left\{ \left( x_j,x_j \right) \,\, \mid \,\,\lambda _i\ne \lambda _j,\lambda _{i}^{*}=\lambda _{j}^{*},i<j \right\}$  
在C中隶属**不同**簇，在C\*中隶属**不同**簇的样本对集合DD（Different Different）：$d=\left| SS \right|, \left| SS \right|=\left\{ \left( x_j,x_j \right) \,\, \mid \,\,\lambda _i\ne \lambda _j,\lambda _{i}^{*}\ne \lambda _{j}^{*},i<j \right\}$  
由上述定义得到下表中聚类性能度量评价外部指标：  

| 性能度量外部指标 |  表达式（[0,1]内越大越好） |  
| :-------: | :------------: |  
| Jaccard系数 | $JC=\frac{a}{a+b+c}$ |  
| FM系数 | $FMI=\sqrt{\frac{a}{a+b}\cdot \frac{a}{a+c}}$ |  
| Rand系数 | $RI=\frac{2\text{（}a+d\text{）}}{m\left( m-1 \right)}$ |       

&nbsp; 
+ 内部指标（internal index）  
当聚类划分的簇为$\boldsymbol{C}=\left\{ C_1,C_2,...,C_k \right\}$，定义：  
样本之间的距离（2-范数/欧式距离）：$$dist\left( \boldsymbol{x}_{\boldsymbol{i}},\boldsymbol{x}_{\boldsymbol{j}} \right) =||\boldsymbol{x}_i-\boldsymbol{x}_j||_2=\sqrt{|\boldsymbol{x}_{i1}-\boldsymbol{x}_{j1}|^2+|\boldsymbol{x}_{i2}-\boldsymbol{x}_{j2}|^2+...+|\boldsymbol{x}_{in}-\boldsymbol{x}_{jn}|^2}$$ 
簇的中心点：$$\mu =\frac{1}{|C|}\sum_{i=1}^{|C|}{x_i}$$
簇C内样本间平均距离：$$avg\left( C \right) =\frac{2}{|C|\left( |C|-1 \right)}\sum{dist\left( \boldsymbol{x}_i,\boldsymbol{x}_j \right)}\,\,   ，1\leqslant i<j\leqslant |C|$$ 
簇C内样本间最远距离：  
$$diam\left( C \right) =\max dist\left( \boldsymbol{x}_i\text{，}\boldsymbol{x}_j \right)  ， 1\leqslant i<j\leqslant |C|$$
簇$C_i$与簇$C_i$最近样本间的距离：$$d_{\min}\left( C_i,C_j \right) =\min dist\left( \boldsymbol{x}_i,\boldsymbol{x}_j \right) ，    x_i\in C_i,x_j\in C_j$$  
簇$C_i$与簇$C_i$中心点之间的距离：$$d_{cen}\left( C_i,C_j \right) =dist\left( \boldsymbol{\mu }_i,\boldsymbol{\mu }_j \right)$$     
由上述定义得到下表中聚类性能度量评价内部指标： 

| 性能度量内部指标 |  表达式  |  
| :-------: | :------------: |
|DB指数(越小越好)| $DBI=\frac{1}{k}\sum_{i=1}^k{\underset{j\ne i}{\max}\left( \frac{avg\left( C_i \right) +avg\left( C_j \right)}{d_{cen}\left( C_i,C_j \right)} \right)}$|
|Dunn指数(越大越好)|$DI=\underset{1\leqslant i\leqslant k}{\min}\left\{ \underset{j\ne i}{\min}\left( \frac{d_{\min}\left( C_i,C_j \right)}{\underset{1\leqslant l\leqslant k}{\max}\,\,diam\left( C_l \right)} \right) \right\}$|     

&nbsp; 
## 9.3 距离度量
$dist\left( \cdot ,\cdot \right)$  是一种距离度量（distance measure）,需要满足非负性、同一性、对称性、直递性。  
+ 有序属性距离度量——闵可夫斯基距离（minkowski distance）

| p 取值 | 名称 |  表达式  |  
| :-------: | :------------: | :------------------------------------------------------: |
| $p\geqslant 1$ |闵可夫斯基距离   |   $dist_{mk}\left( \boldsymbol{x}_i,\boldsymbol{x}_j\right)=\left( \sum_{u=1}^n{\|x_{iu}-x_{ju}\|^p} \right) ^{\frac{1}{p}}$ |
|$p=1$|曼哈顿距离|$dist_{man}\left( \boldsymbol{x}_{\boldsymbol{i}},\boldsymbol{x}_{\boldsymbol{j}} \right) =\|\|\boldsymbol{x}_i-\boldsymbol{x}_j\|\|_1=\|x_{i1}-x_{j1}\|+\|x_{i2}-x_{j2}\|+...+\|x_{in}-x_{jn}\|$ |
|$p=2$|欧式距离(多维向量常用)|$dist_{ed}\left( \boldsymbol{x}_{\boldsymbol{i}},\boldsymbol{x}_{\boldsymbol{j}} \right) =\|\|\boldsymbol{x}_i-\boldsymbol{x}_j\|\|_2=\sqrt{\|x_{i1}-x_{j1}\|^2+\|x_{i2}-x_{j2}\|^2+...+\|x_{in}-x_{jn}\|^2}$|

&nbsp; 
+ 无序属性距离度量——Value Different Metric(VDM)  
属性*u*上取值为*a*的样本数量定义为 $m_{u,a}$  
在第i个样本簇中，在属性*u*上取值为*a*的样本数量定义为 $m_{u,a,i}$  
样本簇数量为k
在属性*u*上两个离散值*a*,*b*之间的VDM距离为：
$$VDM_p\left( a,b \right) =\sum_{i=1}^k{|\frac{m_{u,a,i}}{m_{u,a}}-\frac{m_{u,b,i}}{m_{u,b}}|^p}$$  

+ 混合属性距离度量——结合闵可夫斯基距离和VDM距离，假设样本集有$n_{c}$个有序属性，$n-n_{c}$个无序属性，则有：
$$MinkovDM_p\left( \boldsymbol{x}_{\boldsymbol{i}},\boldsymbol{x}_{\boldsymbol{j}} \right) =\left( \sum_{u=1}^{n_c}{|x_{iu}-x_{ju}|^p}+\sum_{u=n_c+1}^n{VDM_p\left( x_{iu}-x_{ju} \right)} \right) ^{\frac{1}{p}}$$

## 9.4 原型聚类
原型聚类（基于原型的聚类，prototype-based clustering）假设聚类结构可以通过一组原型刻画。原型聚类算法一般先对原型初始化，然后对原型进行更新迭代求解。以下几种为著名的原型聚类算法。  
+ **k 均值算法（k-means）**  
k均值算法的优化目标是最小化样本集的平方误差：
$$E=\sum_{i=1}^k{\sum_{x_i\in C_i}{||\boldsymbol{x}-\boldsymbol{\mu }_i||_{2}^{2}}}$$
平方误差$E$描述了簇内样本围绕均值向量$\boldsymbol{\mu }_i$的紧密程度，平方误差$E$越小，簇内样本越精密。最小化平方误差$E$需要考察所有可能的簇划分（NP难问题），另一可行的方法是对均值向量初始化，采用贪心策略进行迭代优化，直到满足要求。


























