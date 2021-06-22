# Matrix

## 加减法
要求矩阵同型
$$\mathbf{A}+\mathbf{B}=\mathbf{B}+\mathbf{A}$$
$$\left( \mathbf{A}+\mathbf{B} \right) +\mathbf{C}=\mathbf{A}+\left( \mathbf{B}+\mathbf{C} \right)$$
$$\mathbf{A}-\mathbf{B}=\mathbf{A}+\left( -\mathbf{B} \right)$$  

&nbsp;
## 数乘
$$\left( \lambda \mu \right) \mathbf{A}=\lambda \left( \mu \mathbf{A} \right) $$
$$\left( \lambda +\mu \right) \mathbf{A}=\lambda \mathbf{A}+\mu \mathbf{A}$$
$$\lambda \left( \mathbf{A}+\mathbf{B} \right) =\lambda \mathbf{A}+\lambda \mathbf{B}$$

&nbsp;
## 矩阵相乘
$$\left( \mathbf{AB} \right) \mathbf{C}=\mathbf{A}\left( \mathbf{BC} \right)$$
$$\mathbf{\lambda }\left( \mathbf{AB} \right) =\left( \mathbf{\lambda A} \right) \mathbf{B}=\mathbf{A}\left( \mathbf{\lambda B} \right) $$
$$\mathbf{A}\left( \mathbf{B}+\mathbf{C} \right) =\mathbf{AB}+\mathbf{AC}$$
$$\left( \mathbf{B}+\mathbf{C} \right) \mathbf{A}=\mathbf{BA}+\mathbf{CA}$$

&nbsp;
## 转置
$$\left( \mathbf{A}^{\mathrm{T}} \right) ^{\mathrm{T}}=\mathbf{A}$$
$$\left( \mathbf{A}+\mathbf{B} \right) ^{\mathrm{T}}=\mathbf{A}^{\mathrm{T}}+\mathbf{B}^{\mathrm{T}}$$
$$\left( \lambda \mathbf{A} \right) ^{\mathrm{T}}=\lambda \mathbf{A}^{\mathrm{T}}$$
$$\left( \mathbf{AB} \right) ^{\mathrm{T}}=\mathbf{B}^{\mathrm{T}}\mathbf{A}^{\mathrm{T}}$$

&nbsp;
## 行列式
$\mathbf{A，B}$为$n$阶方阵
$$\left| \mathbf{A}^{\mathrm{T}} \right|=\left| \mathbf{A} \right|$$
$$\left| \lambda \mathbf{A} \right|=\lambda ^n\mathbf{A}$$
$$\left| \mathbf{AB} \right|=\left| \mathbf{A} \right|\left| \mathbf{B} \right|$$

&nbsp;
## 逆矩阵
若存在（$\mathbf{E}$为单位矩阵）
$$\mathbf{AB}=\mathbf{BA}=\mathbf{E}$$
则称A可逆，B为A的逆矩阵，记作 $\mathbf{B}=\mathbf{A}^{-1}$。
逆矩阵运算定律：
$$\left( \mathbf{A}^{-1} \right) ^{-1}=\mathbf{A}$$
$$\left( \lambda \mathbf{A} \right) ^{-1}=\frac{1}{\lambda}\mathbf{A}^{-1}$$
$$\left( \mathbf{AB} \right) ^{-1}=\mathbf{B}^{-1}\mathbf{A}^{-1}$$

&nbsp;
## 线性相关
若存在一组数$k_1,k_2,...,k_m$, 使得向量组$\mathbf{A}$=（$a_1,a_2,...,a_m$）中存在:
$$k_1a_1+k_2a_2+...+k_ma_m=0$$
则称向量组$\mathbf{A}$ 是线性相关的，否则线性无关。  
若存在一组数$k_1,k_2,...,k_m$：
$$k_1a_1+k_2a_2+...+k_ma_m=\boldsymbol{b}$$
则称向量$\boldsymbol{b}$ 能由向量组$\mathbf{A}$ 线性表示。

&nbsp;
## 秩
定义一：秩为矩阵非零子式的最高阶数。非零子式：矩阵中任取k行k列，位于这些行列交叉处的$k^2$个元素按原始位置组成的k阶行列式。  
下图为根据定义一计算秩的例子：  
![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210611135654.png)

定义二：矩阵A中，能在A中选出 r 个向量$a_1,a_2,...,a_r$，满足：
$$a_1,a_2,...,a_r线性无关，且任意r+1个向量都线性相关$$
即矩阵中所有行向量中极大线性无关组的元素个数（方程组中真正有用的方程个数）。

&nbsp;
## 基
设V为向量空间，若存在 r 个向量 $a_1,a_2,...,a_r$ ，满足：
$$\left( 1 \right) \,\,a_1,a_2,...,a_r\text{线性无关}$$
$$\left( 2 \right) \,\,V\text{中任意向量都可由}a_1,a_2,...,a_r\text{线性表示}$$
则称 $a_1,a_2,...,a_r$ 为向量空间V的一个基，r为向量空间的维数。
在V中任意向量可表示为：
$$x=k_1a_1+k_2a_2+...+k_ra_r$$
其中 $k_1,k_2,...,k_r$ 称为 $x$ 的坐标。

&nbsp;
## 标准正交基
若向量$\boldsymbol{a},\boldsymbol{b}$内积为0，即：
$$\left< \boldsymbol{a},\boldsymbol{b} \right> =0$$
则称向量$\boldsymbol{a},\boldsymbol{b}$正交。  
若向量$\boldsymbol{a},\boldsymbol{b}$满足正交关系且都为单位向量，则称向量$\boldsymbol{a},\boldsymbol{b}$为标准正交基。
若矩阵A满足：
$$\mathbf{A}^{\mathrm{T}}\mathbf{A}=\mathbf{E}\,\,\left( \text{即}\mathbf{A}^{-1}=\mathbf{A}^{\mathrm{T}} \right)$$
称矩阵A为正交矩阵，列向量都为单位向量，且凉凉正交。

&nbsp;
## 特征值
若 A 为 n 阶矩阵，数 $\lambda$ 和 n 维向量 x 满足：
$$\mathbf{A}\boldsymbol{x}=\lambda \boldsymbol{x}$$
则称数 $\lambda$ 为 A 的特特征值， x 为 $\lambda$ 对应的特征向量。  
下图为计算矩阵 A 特征值及对应特征向量的例子：

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210611145950.png)

