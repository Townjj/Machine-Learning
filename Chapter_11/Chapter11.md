# Chapter 11 特征选择和稀疏学习

## 1.子集搜索与评价  
一个样本通常有多个属性，如西瓜有色泽，根蒂，颜色等。将属性称之为特征，对一个学习任务而言，有用的特征称之为“相关特征”（relevant feature）,用处不大的特征称之为“无关特征”（irrelevant feature）。在特征集合中选出相关特征子集的过程称之为特征选择。特征选择不仅可以解决维数灾难，亦可以降低学习任务的难度。    
### 子集搜索（subset search）   
子集搜索分为前向(forward)搜索，后向(backward)搜索和双向(bidirectional)搜索。采用贪心的策略，考虑每一轮选择中都选择到了最优的特征子集。  
前向搜索的算法如下：  

    Input:  
        特征集$\left\{ a_1,a_2,...,a_{\mathrm{d}} \right\}$
    Do:
        将每个特征视为一个属性子集
    Repeat：
        在k-1轮的最优属性集中每次放入一个未被选择特征，构成包含有k个特征的候选特征子集（可生成d-k个）
        评价d-k个候选特征子集
        选取最优的一个特征子集（包含有k个特征）
        Until:
            第k+1轮加入一个新特征后的最优子集不如第k轮最优子集效果好
    Output:
        最优特征子集

### 子集评价（subset evaluation）
对每个特征，使用信息增益进行评价后选择最优特征。  
假定数据集D中第i类样本所占比例为 $p_i$ ，对属性子集A，当前特征有V个取值时，根据样本取值将样本集D分成了V个子集$\left\{ D^1,D^2,...,D^V \right\}$，每个子集中样本在属性子集A中取值相同，计算属性子集A的信息增益为：
$$Gain\left( A \right) =Ent\left( D \right) -\sum_{v=1}^V{\frac{\left| D^V \right|}{\left| D \right|}}Ent\left( D^V \right)$$
其中信息熵定义为：
$$Ent\left( D \right) =-\sum_{k=1}^{\left| y \right|}{p_k\log _2}p_k$$
其中$|y|$表示样本集D在属性集A中特征类别的数量。
信息增益$Gain\left( A \right)$越大，则特征子集A包含有助于分类的信息越多，对每个候选特征子集D进行评价，选取信息增益最大的候选特征子集作为最优特征子集。
    
  
&nbsp;  
## 特征选择方法 = 特征子集搜索机制 + 子集评价机制
常见特征选择方法可以分为：过滤式（filter）、包裹式（wrapper）、嵌入式（embedding）


&nbsp;  
## 2.过滤式选择
过滤式选择方法先对数据集进行特征选择，然后再训练学习器，特征选择的过程与学习器无关。  
Relif(Relevant feature)算法设定“相关统计量”来度量特征重要性，相关统计量（向量）的每个分量对应一个特征，特征子集的重要性由子集中特征对应的相关统计量分量之和来决定。可设置阈值或欲选取的特征个数来筛选最优特征。
Relif算法流程如下：
![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210615222320.png)

容易看出，$\delta ^j$越大，对应属性的分类能力就越强。

&nbsp;  
## 3.包裹式选择  
包裹式选择直接将学习器的性能作为特征子集的评价指标，给学习器量身定做最优特征子集。  
**LVW (Las Vegas Wrapper)** 使用随机策略进行子集搜索，将最终分类器（学习器）的误差作为特征子集的评价指标。
先随机产生特征子集A，通过交叉验证估计学习误差，不断更新（误差更小的特征子集）或（误差相等但子集内特征数更少的特征子集）作为最优子集，直到连续 T 轮不再更新。

&nbsp;  
## 4.嵌入式选择与L1正则化
嵌入式选择将特征选择与学习器训练融合在一起，共同在同一个优化过程中完成。
在简单的线性回归模型中，以平方误差为损失函数时，当样本特征过多容易过拟合，加入$L_2$范数正则化项后的优化目标为：  
$$\underset{\boldsymbol{\omega }}{\min}\,\,\sum_{i=1}^m{\left( y_i-\boldsymbol{\omega }^{\mathrm{T}}x_i \right) ^2+\lambda ||\boldsymbol{\omega }||_{2}^{2}}$$ 
其中正则化系数$\lambda >0$，上式称为**岭回归（ridge regression）**。  
将$L_2$范数正则化项推广为$L_p$范数正则化项，当p=1时，采用$L_1$范数正则化项的优化目标为：  
$$\underset{\boldsymbol{\omega }}{\min}\,\,\sum_{i=1}^m{\left( y_i-\boldsymbol{\omega }^{\mathrm{T}}x_i \right) ^2+\lambda ||\boldsymbol{\omega }||_{1}^{\,\,}}$$其中正则化系数$\lambda >0$，上式称为LASSO（Least Absolute Shrinkage and Selection Operator,最小绝对收缩选择算子）。
$L_1$范数正则化及$L_2$范数正则化都可以降低过拟合风险，但$L_1$范数正则化的突出优势是更易获得稀疏解，即求得的向量$\boldsymbol{\omega }$将含有更多的零分量，零向量对应的特征被剔除。换言之，基于$L_1$范数正则化的学习方法是一种嵌入式特征选择方法，其特征选择过程与学习器训练过程融为一体，共同完成。
![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/PGD.png)