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





