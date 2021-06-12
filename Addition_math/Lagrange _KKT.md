# 拉格朗日乘子法与KKT条件

##　梯度　　
在单变量函数中，梯度是函数的微分，代表着函数在某个给定点的切线斜率。
在多变量函数中，梯度是空间内的一个向量，梯度的方向是函数在给定点上升或下降最快的方向。　　
下图为梯度计算的例子：　　
![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210612103756.png)

&nbsp;
##　拉格朗日乘子法
拉格朗日乘子法是一种解决约束优化的算法。通过引入拉格朗日乘子，将一个含有 k 个变量和 m 个约束条件的约束优化问题转化为含有 k+m 个变量的无约束优化问题。　　  

在$f\left( x \right)$的等高线中其一与$\,h_i\left( x \right) ＝0$相切时，存在最优解，此时相切点梯度反向。$\,h_i\left( x \right) ＝0$上的点即为 ｘ 的可行域, 如下图：   　　

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210612160833.png)

设目标函数为$f\left( x \right)$　，约束条件为$\,h_k\left( x \right)$，问题描述为　：
$$\min  f\left( x \right)$$
$$s.t.\ \ h_{\mathrm{i}}\left( x \right) =0  \ \ \  \left( \mathrm{i}=1,2,...,\mathrm{m} \right)$$  

定义拉格朗日函数：
$$LL\left( x,\lambda \right) =f\left( x \right) +\sum_{\mathrm{i}=1}^{\mathrm{m}}{\lambda _{\mathrm{i}}h_{\mathrm{i}}\left( x \right)}$$
其中$\lambda _l$为约束条件对应的拉格朗日乘子。
要求得最优解，则对拉格朗日函数各个变量求偏导数置零（LL梯度为零）：
$$\frac{\partial F}{\partial x_i}=0$$
$$\frac{\partial F}{\partial \lambda _i}=0$$
即可求得最优解。
以下为拉格朗日函数构建示例：
![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210612153343.png)


&nbsp;
## KKT条件 

在使用拉格朗日乘子法时，当约束条件中有不等式约束时，通过拉格朗日乘子法解出最优值时，最优值满足KKT条件。  
$$\min  f\left( x \right)$$

$$s.t. h_{\mathrm{i}}\left( x \right) =0   \left( \mathrm{i}=1,2,...,\mathrm{m} \right)$$

$$\mathrm{g}_{\mathrm{j}}\left( x \right) \leqslant 0   \left( \mathrm{j}=1,2,...,\mathrm{n} \right)$$
定义拉格朗日函数为：
$$LL\left( \boldsymbol{x},\boldsymbol{\lambda },\boldsymbol{\mu } \right) =f\left( \boldsymbol{x} \right) +\sum_{\mathrm{i}=1}^{\mathrm{m}}{\lambda _{\mathrm{i}}h_{\mathrm{i}}\left( \boldsymbol{x} \right)}+\sum_{\mathrm{j}=1}^{\mathrm{n}}{\mathrm{\mu}_{\mathrm{j}}\mathrm{g}_{\mathrm{j}}\left( \boldsymbol{x} \right)}$$
则优化问题的极值点一定满足KKT条件，即：
$$\mathrm{g}_{\mathrm{j}}\left( \boldsymbol{x} \right) \leqslant 0$$
$$\mathrm{\mu}_{\mathrm{j}}\geqslant 0$$
$$\mathrm{\mu}_{\mathrm{j}}\mathrm{g}_{\mathrm{j}}\left( \boldsymbol{x} \right) =0$$  
