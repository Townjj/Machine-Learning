# Kernel Function / Kernel Trick
核函数在机器学习中应用范围广泛，包括SVM，LDA，LR，KPCA等算法。在样本构成的低维样本空间中，样本集往往是线性不可分的，核方法将低维空间中的样本通过映射到更高维度的空间甚至是无穷维度空间内以寻找线性可分的超平面。  

## 定义

在希尔伯特空间$\boldsymbol{H}$内，如果存在一个从空间$\boldsymbol{\chi }$到$\boldsymbol{H}$的映射$\phi \left( \cdot \right)$，使得空间 $\boldsymbol{\chi }$ 内所有的 $\boldsymbol{x}_i,\boldsymbol{x}_j$ ，函数$K\left( \boldsymbol{x}_i,\boldsymbol{x}_j \right)$都满足条件：  
$$K\left( \boldsymbol{x}_i,\boldsymbol{x}_j \right) =<\phi \left( \boldsymbol{x}_i \right) ,\phi \left( \boldsymbol{x}_j \right) >=\phi \left( \boldsymbol{x}_i \right) \cdot \phi \left( \boldsymbol{x}_j \right)$$ 
$\phi \left( \cdot \right)$为映射函数，$\,\,K\left( \cdot ,\cdot \right)$为核函数。$\phi \left( \boldsymbol{x}_i \right) \cdot \phi \left( \boldsymbol{x}_j \right)$为$\boldsymbol{x}_i,\boldsymbol{x}_j$的内积（点积）。

