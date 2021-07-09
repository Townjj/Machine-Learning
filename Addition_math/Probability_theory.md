# Probability theory

## 常用公式  
条件概率  
$$P\left( B|A \right) =\frac{P\left( AB \right)}{P\left( A \right)}$$
$$P\left( A|B \right) =\frac{P\left( AB \right)}{P\left( B \right)}$$  
乘法公式（可由条件概率推出）  
$$P\left( AB \right) =P\left( B|A \right) P\left( A \right) =P\left( A|B \right) P\left( B \right)$$
全概率公式  
$$P\left( B \right) =\sum_{j=1}^n{P\left( A_j \right) P}\left( B|A_j \right)$$  
贝叶斯公式
$$P\left( A_j|B \right) =\frac{P\left( B|A_j \right)}{P\left( B \right)}=\frac{P\left( A_j \right) P\left( B|A_j \right)}{\sum_{j=1}^n{P\left( A_j \right) P}\left( B|A_j \right)}$$  


&nbsp;
## 高斯分布  
数据 x 是一维单高斯模型分布时，高斯分布遵从如下概率密度函数：
$$P\left( x|\mu ,\sigma \right) =\frac{1}{\sqrt{2\pi \sigma ^2}}\exp \left( -\frac{\left( x-\mu \right) ^2}{2\sigma ^2} \right)$$
其中$\mu$为数据均值，$\sigma$为数据标准差。  

数据x是多维数据时，且服从单高斯分布，具有以下概率密度函数：
$$P\left( x|\mu ,\varSigma \right) =\frac{1}{\left( 2\pi \right) ^{\frac{D}{2}}|\varSigma |^{\frac{1}{2}}}\exp \left( -\frac{\left( x-\mu \right) ^T\varSigma ^{-1}\left( x-\mu \right)}{2} \right)$$  
其中$\mu$为数据均值，$\varSigma$为协方差矩阵，D为数据维度。  

混合高斯分布是用多个高斯（正态）分布去模拟概率分布，由于高斯分布优秀的数学性质，使得高斯混合分布的应用广泛，高斯混合分布的概率密度为：  
$$p\left( x \right) =\sum_{i=1}^N{\alpha _i\cdot}P\left( x|\mu _i,\varSigma _i \right)$$  
$\alpha _i$为数据属于第i个子模型的概率，和为1。


&nbsp;
## 极大似然估计
极大似然估计利用已知样本结果信息，反推最具可能导致这些样本结果出现的模型参数值。对于正态分布$P\left( x|\mu ,\sigma \right)$，如果x未知，$\mu ,\sigma$已知，则P为概率函数；
若$\mu ,\sigma$未知，x已知，则P为似然函数。似然函数描述了对于不同的函数模型，出现样本点x的概率是多少。求解时令最大似然函数导数为0，求得对应的模型参数。