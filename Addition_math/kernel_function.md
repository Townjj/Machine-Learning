# Kernel Function / Kernel Trick
核函数在机器学习中应用范围广泛，包括SVM，LDA，LR，KPCA等算法。在样本构成的低维样本空间中，样本集往往是线性不可分的，核方法将低维空间中的样本通过映射到更高维度的空间甚至是无穷维度空间内以寻找线性可分的超平面。  
&nbsp;
## 1 定义

在希尔伯特空间$\boldsymbol{H}$内，如果存在一个从空间$\boldsymbol{\chi }$到$\boldsymbol{H}$的映射$\phi \left( \cdot \right)$，使得空间 $\boldsymbol{\chi }$ 内所有的 $\boldsymbol{x}_i,\boldsymbol{x}_j$ ，函数$K\left( \boldsymbol{x}_i,\boldsymbol{x}_j \right)$都满足条件：  
$$K\left( \boldsymbol{x}_i,\boldsymbol{x}_j \right) =<\phi \left( \boldsymbol{x}_i \right) ,\phi \left( \boldsymbol{x}_j \right) >=\phi \left( \boldsymbol{x}_i \right) \cdot \phi \left( \boldsymbol{x}_j \right)$$ 
$\phi \left( \cdot \right)$为映射函数，$\,\,K\left( \cdot ,\cdot \right)$为核函数。$\phi \left( \boldsymbol{x}_i \right) \cdot \phi \left( \boldsymbol{x}_j \right)$为$\boldsymbol{x}_i,\boldsymbol{x}_j$的内积（点积）。     
&nbsp;
## 2 计算
在机器学习中，我们大部分仅需使用样本间的内积用以距离度量，而不关心高维空间内的样本及映射函数的具体形式。
核函数的引入，使得样本的内积可以在低维空间直接计算高维空间中的样本内积得出，且不必得到映射函数，使得样本的内积计算容易，避免维数灾难。  
关于核函数的便利性，更形象的表达见下图：
&nbsp; 

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210609102614.png)
&nbsp;
## 3 类型
核函数的类型有很多，“核函数选择”成为一些机器学习算法（SVM等）的最大变数，若核函数选择不合适，则意味着样本将被映射到一个不合适的特征空间中。  
西瓜书中列举的核函数如下图： 

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210609103645.png)

