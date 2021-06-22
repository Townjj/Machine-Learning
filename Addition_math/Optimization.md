# Optimization   最优化

最优化：建模、求极值
最优化问题的标准形式：
$$\underset{\boldsymbol{x}\in R^n}{\min}\,\,f\left( \boldsymbol{x} \right)$$
$$s.t.    g_u\left( \boldsymbol{x} \right) \leqslant 0 ,u=1,2,...m$$
$$h_v\left( \boldsymbol{x} \right) =0,v=1,2,...p\left( p<n \right)$$

线性规划：目标函数和约束条件都为线性函数
非线性规划：目标函数和约束条件不都为线性函数  

目标函数等值线：具有相同目标函数值的设计点（决策变量）构成的平面曲线。当决策变量大于2时，为等值面，超曲面。  
用平行于x1x2平面的平面截取f(x)，界面边界在x1x2平面的投影即为其中一条等值线。如下图：   

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622194602.png)  
  


&nbsp;  
可行域：受约束函数的限制，目标函数的可行域可表示为多个不等式约束函数$g_u\left( \boldsymbol{x} \right)$围成的区域，且当有等式约束时，可行域为$h_v\left( \boldsymbol{x} \right)$上在不等式约束函数围成区域内的点。  
可行域的确定见下图（当等式约束数量p大于决策变量数量n时，不一定有可行域，因为多条直线必要穿过同一点）：  

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622195445.png)
  


&nbsp;   
梯度
在单变量函数中，梯度是函数的微分，代表着函数在某个给定点的切线斜率。  
在多变量函数中，梯度是空间内的一个向量，梯度的方向是函数在给定点局部上升或下降最快的方向，相当于一元函数的一次导 数。   
海森矩阵（Hessian Matrix）
海森矩阵相当于多元函数求二次导数组成的矩阵，由于$\frac{\partial ^2f}{\partial x_1\partial x_2}$=$\frac{\partial ^2f}{\partial x_2\partial x_1}$，所以海森矩阵是对称矩阵，计算时只需算下三角或上三角。  
n元函数的梯度、海森矩阵的定义及计算示例见下图：  

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622202854.png)


泰勒展开
泰勒展示式在函数某一点附近用简单多项式近似复杂函数，带来了分析计算上的便利。  
一元函数在$x_0$处的的泰勒式展开为：  
$$f\left( x \right) =f\left( x_0 \right) +f'\left( x_0 \right) \left( x-x_0 \right) +\frac{1}{2!}f''\left( x_0 \right) \left( x-x_0 \right) ^2...+\frac{1}{n!}f^n\left( x_0 \right) \left( x-x_0 \right) ^n+C$$
多元函数在$\boldsymbol{x}_0$处的泰勒级数展开（只取前两项）为：  
$$f\left( \boldsymbol{x} \right) \approx f\left( \boldsymbol{x}_0 \right) +\nabla f\left( \boldsymbol{x} \right) ^{\mathrm{T}}\left( \boldsymbol{x}-\boldsymbol{x}_0 \right) \nabla f\left( \boldsymbol{x} \right) +\frac{1}{2}\left( \boldsymbol{x}-\boldsymbol{x}_0 \right) ^{\mathrm{T}}H\left( \boldsymbol{x} \right) \left( \boldsymbol{x}-\boldsymbol{x}_0 \right)$$
其中$H\left( \boldsymbol{x} \right)$为对于x0处的海森矩阵。  
 

正定、半正定、负定矩阵   
在线性代数中，正定矩阵（positive definite matrix）对应复数中的正实数。半正定矩阵（positive semi-define matrix）对应复数中的非负实数，负定矩阵（negative definite matrix）对于复数中的负数。  
**正定矩阵的定义：** 若 n 阶实对称方阵 A ，对于任意的 n 维非零向量 x , 满足 $\boldsymbol{x}^{\mathrm{T}}\boldsymbol{Ax}>0$ ，则方阵 A 为正定矩阵。  
**半正定矩阵的定义：** 若 n 阶实对称方阵 A ，对于任意的 n 维向量 x , 满足 $\boldsymbol{x}^{\mathrm{T}}\boldsymbol{Ax}\geqslant 0$ ，则方阵 A 为半正定矩阵。  
**正定矩阵的判断：** n 阶方阵 A 的各阶顺序主子式（左上角K阶方阵构成的行列式）均大于0。  
**负定矩阵的判断：** n 阶方阵 A 的各阶顺序主子式（左上角K阶方阵构成的行列式）负正相间（奇数阶小于0，偶数阶大于0）。
不定矩阵的判断：非正定非负定即为不定矩阵。
协方差矩阵要求是半正定的。     
  


无约束条件的极值条件  
在一元函数中，通过令$f'\left( x \right) =0$得到极值点x0，再通过考察$f''\left( x_0 \right)$的正负性判断其为极大点、极小点或拐点。  
类似的，在多元函数中，令$f\left( \boldsymbol{x} \right)$的一阶导数$\nabla f\left( \boldsymbol{x} \right)=0$解出极值点$\boldsymbol{x}_0$，再通过考察$f\left( \boldsymbol{x} \right)$二阶导数$H\left( \boldsymbol{x} \right)$是否为正定判断其为极大点、极小点或鞍点（如x1上极大，x2上极小）。    
一元函数及多元函数的极值条件对比表如下图：  

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622214023.png)


   

多元函数极值求解示例如下图：   

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622221846.png)



 
   

凸集与凸函数  


一元凸函数定义：若函数 f(x) 曲线上任意两点的连线用在 f(x) 曲线上方，则 f(x) 为**凸函数** 。  
在几何中，存在关系$f\left( \boldsymbol{x} \right) \leqslant Y$  

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622224755.png)




凸集定义：集合D中任意两点$\boldsymbol{x}_1$、$\boldsymbol{x}_2$的连线仍在D内，则D内凸集，否则为非凸集。  
多元凸函数定义：f(x)定义域为凸集，对任意实数$0\leqslant \lambda \leqslant 1$，，D中任意两点$\boldsymbol{x}_{\boldsymbol{1}}\text{、}\boldsymbol{x}_2$均存在：
$${\ f\left[ \lambda \boldsymbol{x}_{\boldsymbol{1}}+\left( 1-\lambda \right) \boldsymbol{x}_{\boldsymbol{2}} \right] \leqslant \lambda f\left( \boldsymbol{x}_{\boldsymbol{1}} \right) +\left( 1-\lambda \right) f\left( \boldsymbol{x}_{\boldsymbol{2}} \right) \,\,\ \   \      }0\leqslant \lambda \leqslant 1$$

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622225639.png)
多元凸函数的判定条件为：$f\left( \boldsymbol{x} \right)$的海森矩阵为半正定矩阵。


凸规划
对于优化问题
$$\underset{\boldsymbol{x}\in R^n}{\min}\,\,f\left( \boldsymbol{x} \right)$$
$$s.t.    g_u\left( \boldsymbol{x} \right) \leqslant 0 ,u=1,2,...m$$
当f(x),g(x)均为凸函数时，优化问题为凸规划问题。  
凸规划问题的性质：

* f(x)等值线呈大圈套小圈的形式
* 可行域为凸集
* 局部最小点一定是全局最优解　　
  
   


　　　　　　　　　　　


## 约束优化问题的极值条件　　　　

库恩塔克条件(Kuhn-Tucker conditions，KT条件)是确定某点为极值点的必要条件。如果所讨论的规划是凸规划，那么库恩-塔克条件也是充分条件。  
并非所有约束条件都起作用，即并非落在所有约束条件的边界交集处。对于起作用的边界条件，满足边界函数g(x)或h(x)为0.

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622234120.png)

