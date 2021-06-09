# Chapter 10 降维与度量学习
## 10.1 k 临近学习 
k 邻近（k-Nearest Neighbor,kNN）学习是常见的监督学习方法，给定测试样本，基于某种距离度量找到与测试样本最靠近的k个邻居，在分类任务中基于k个邻居的类别标记使用“投票法”输出测试样本的类别标记预测，在回归任务中使用“平均法”基于k个邻居的实值输出平均值作为测试样本的预测。  
k临近学习基于一个重要假设：任意测试样本 $x$ 附近任意小的 $\delta $ 距离范围内总能找到一个训练样本，即训练样本需要进行较大的采样密度（密采样，dense sample）。引出一个问题：高维下的密采样及距离计算不再容易，引起维数灾难（curse of dimensionality），需要降维处理（dimension reduction）。  
## 10.2 低维嵌入
大多时候数据样本虽然时高维的，但与学习任务相关的仅是某个低维分布，即高维空间内的一个低维嵌入（embedding）。  
降维（dimension reduction）通过数学变换使得高维度的属性空间变为一个低维“子空间”（subspace）使得样本密度增大，距离计算变得容易。  
+  多维缩放降维方法（Multiple Dimensional Scaling，MDS）  
假设样本集有m个d维样本，在原始样本空间的距离矩阵为$\mathbf{D}\in \mathrm{R}^{m\times m}$，D中元素$dist_{ij}$表示样本$\boldsymbol{x}_{i\,\,}$到样本$\boldsymbol{x}_j$的距离。目标是找到降维后的样本集 $\mathbf{Z}\in \mathrm{R}^{d'\times m}$。
 设矩阵$\mathbf{B}=\mathbf{Z}^{\mathrm{T}}\mathbf{Z}\in \mathrm{R}^{m\times m}$为降维后样本集的内积矩阵。  

>由任意两个样本在$d'$空间内的欧式距离等于原始空间内的欧式距离：  
 $$dist_{ij}=||\boldsymbol{z}_i-\boldsymbol{z}_j||$$ 
 有
 $$dist_{ij}^{2}=||\boldsymbol{z}_i-\boldsymbol{z}_j||^2=||\boldsymbol{z}_i||^2+||\boldsymbol{z}_j||^2-2\boldsymbol{z}_{i}^{T}\boldsymbol{z}_j=b_{ii}+b_{jj}-2b_{ij}$$
 将样本集Z去中心化（$\sum_{i=1}^m{\boldsymbol{z}_i}=0$）可得： 
 $$b_{ij}=-\frac{1}{2}\left( dist_{ij}^{2}-dist_{i\cdot}^{2}-dist_{\cdot j}^{2}+dist_{\cdot \cdot}^{2} \right)$$
 其中$$dist_{i\cdot}^{2}=\frac{1}{m}\sum_{j=1}^m{dist_{ij}^{2}}$$
$$dist_{\cdot j}^{2}=\frac{1}{m}\sum_{i=1}^m{dist_{ij}^{2}}$$
$$dist_{\cdot \cdot}^{2}=\frac{1}{m^2}\sum_{i=1}^m{\sum_{j=1}^m{dist_{ij}^{2}}}$$
到此已从距离矩阵D得到降维后样本集Z的内积矩阵B，接下来对矩阵B做特征值分解：
$$\mathbf{B}=\mathbf{V\Lambda V}^T$$
其中$\mathbf{V}$ 为特征向量矩阵，$\mathbf{\Lambda }$为特征值对角矩阵：
$$\mathbf{\Lambda }=\mathrm{diag}\left( \lambda _1,\lambda _2,...,\lambda _d \right) \,\, ,(\lambda _1\geqslant \lambda _2\geqslant ...\geqslant \lambda _d)$$
现实用只需降维后空间与原始空间内的样本距离尽可能接近，可选取$d'$个最大非零特征值（$d'\gg d$），构成特征值对角矩阵$\overset{\sim}{\mathbf{\Lambda }}$，及其对应特征向量矩阵$\overset{\sim}{\mathbf{V}}$，可得到降维后的样本集：
$$\mathbf{Z}=\overset{\sim}{\mathbf{\Lambda }}^{\frac{1}{2}}\overset{\sim}{\mathbf{V}^T}\text{，}\mathbf{Z}\in \mathrm{R}^{d'\times m}$$  
至此，低维子空间$\mathbf{Z}$已推导完成。  

欲获得低维子空间，最简单的办法是对原始高维空间进行线性变换，原始高维样本空间$\mathbf{X}$包含m个d维样本，通过包含变换矩阵$\mathbf{W}$（$\mathbf{W}\in \mathrm{R}^{d\times d'}$），可得到降维后的具有m个$d'$维样本的样本空间$\mathbf{Z}$，过程如下图：

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210609104148.png)

&nbsp; 
## 10.3 主成成分分析
主成成分分析（Principal Component Analysis,PCA）是常用的一种降维方法，要求降维后的样本空间对样本具有最大可分性，并用一个超平面对所有样本进行表达。PCA在线性降维时，需要基于最近重构性和最大可分性对变换矩阵$\mathbf{W}$（$\mathbf{W}\in \mathrm{R}^{d\times d'}$）约束。 
>PCA中由最大可分性出发推导降维变换矩阵$\mathbf{W}$（$\mathbf{W}\in \mathrm{R}^{d\times d'}$）  
要使投影后样本点$\text{（}x_i\Rightarrow \mathbf{W}^{\mathrm{T}}x_i\text{）}$具有最大可分性，即样本方差最大。样本协方差矩阵为：
$$\sum_i{\mathbf{W}^{\mathrm{T}}x_ix_{i}^{T}\mathbf{W}}$$  
优化目标为：
$$\underset{\mathbf{W}}{\max}\,\,\mathrm{tr}\left( \mathbf{W}^{\mathrm{T}}\mathbf{XX}^{\mathrm{T}}\mathbf{W} \right)$$
$$s.t.\ \   \mathbf{W}^{\mathrm{T}}\mathbf{W}=\mathbf{I}$$
使用拉格朗日乘子法：
$$\mathbf{XX}^{\mathrm{T}}\boldsymbol{\omega }_i=\lambda _i\boldsymbol{\omega }_i$$
对协方差矩阵$\mathbf{XX}^{\mathrm{T}}$进行特征值分解得到特征值：
$\lambda _1,\lambda _2,...,\lambda _d \ ,\text{（}\lambda _1\geqslant \lambda _2\geqslant ...\geqslant \lambda _d\text{）}$  
取前$d'$个特征值对应的特征向量构成变换矩阵（投影矩阵）$\mathbf{W}^*$:
$$\mathbf{W}^*=\left( \boldsymbol{\omega }_1\text{，}\boldsymbol{\omega }_2\text{，}...\text{，}\boldsymbol{\omega }_{d'} \right)$$
至此已得到高维到低维的转换矩阵$\mathbf{W}^*$，推导完成  

&nbsp; 
## 10.4 核化线性降维
线性降维方法中假设从高维到低维的映射是线性的，不少机器学习任务中需要非线性映射到低维空间中才能找到合适的低维嵌入。将原本采样的低维空间称为本真（intrinsic）低维空间。
基于核函数（核技巧）对线性降维方法进行非线性降维是一种常用方法。  
+ 核组成成分分析算法（Kernel PCA，KPCA）  
KPCA的解决思路是将原始样本空间中的$\boldsymbol{x}_{i\,\,}$通过映射函数$\phi \left( \cdot \right)$映射到高维特征空间中，在高维特征空间中实施PCA算法即可（通过核函数计算内积）。  
由$\mathbf{XX}^{\mathrm{T}}\boldsymbol{\omega }_i=\lambda _i\boldsymbol{\omega }_i$，对于投影矩阵中$\mathbf{W}=\left( \boldsymbol{\omega }_1\text{，}\boldsymbol{\omega }_2\text{，}...\text{，}\boldsymbol{\omega }_{d'} \right)$中的$\boldsymbol{\omega }_j$，有
$$\left( \sum_{i=1}^m{\boldsymbol{z}_i\boldsymbol{z}_{i}^{\mathrm{T}}} \right) \boldsymbol{\omega }_j=\lambda _i\boldsymbol{\omega }_j$$  
其中${z}_{i}$是本真结构中的样本点$\boldsymbol{x}_{i\,\,}$映射到高维特征空间的像，将$\boldsymbol{\omega }_j$（i表示样本序号，j表示第j维）表示为
$$\boldsymbol{\omega }_j=\sum_{i=1}^m{\boldsymbol{z}_i\alpha _{i}^{j}}, \ 其中\alpha _{i}^{j}=\frac{1}{\lambda _j}\boldsymbol{z}_{i}^{\mathrm{T}}\boldsymbol{\omega }_j$$
引入映射函数$\phi \left( \cdot \right)$得到：  
$$\boldsymbol{\omega }_j=\sum_{i=1}^m{\phi \left( \boldsymbol{x}_i \right) \alpha _{i}^{j}}$$
引入核函数$k\left( \boldsymbol{x}_i,\boldsymbol{x}_j \right) =\phi \left( \boldsymbol{x}_i \right) ^{\mathrm{T}}\phi \left( \boldsymbol{x}_j \right)$后化简得到： 
$$\mathbf{K}\boldsymbol{\alpha }^j=\lambda _j\boldsymbol{\alpha }^j$$
其中$\mathbf{K}$为核矩阵，$\left( \mathbf{K} \right) _{ij}=k\left( \boldsymbol{x}_i,\boldsymbol{x}_j \right)$，$\boldsymbol{\alpha }^j=\left( \alpha _{1}^{j};\alpha _{2}^{j},...,\alpha _{m}^{j} \right)$，对核矩阵$\mathbf{K}$进行特征值分解，取前$d'$大的特征值对应的特征向量即可。


&nbsp; 
## 10.5 流形学习



