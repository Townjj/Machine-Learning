# Optimization   最优化
## 引言  
在机器学习中，通常构造出一个带有约束条件的目标函数进行优化，学习最优化方法有助于机器学习算法。本章节借助天津大学网课（https://www.xuetangx.com/course/ecust13051002148/4231479?channel=learn_title）对部分最优化方法归纳总结。  
学习时间：2021/06/21-2021/06/26  
文章结构：  

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/visio_file.png)

&nbsp; 
# 1 概念
## 最优化  
最优化问题的求解步骤为建模、求极值。
最优化问题的标准形式：
$$\underset{\boldsymbol{x}\in R^n}{\min}\,\,f\left( \boldsymbol{x} \right)$$
$$s.t.    g_u\left( \boldsymbol{x} \right) \leqslant 0 ,u=1,2,...m$$
$$h_v\left( \boldsymbol{x} \right) =0,v=1,2,...p\left( p<n \right)$$

&nbsp; 
## 线性/非线性规划
线性规划：目标函数和约束条件都为线性函数  
非线性规划：目标函数和约束条件不都为线性函数  

&nbsp; 
## 目标函数等值线
具有相同目标函数值的设计点（决策变量）构成的平面曲线。当决策变量大于2时，为等值面，超曲面。  
用平行于x1x2平面的平面截取f(x)，界面边界在x1x2平面的投影即为其中一条等值线。如下图：   

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622194602.png)  
  


&nbsp;  
## 可行域
可行域：受约束函数的限制，目标函数的可行域可表示为多个不等式约束函数$g_u\left( \boldsymbol{x} \right)$围成的区域，且当有等式约束时，可行域为$h_v\left( \boldsymbol{x} \right)$上在不等式约束函数围成区域内的点。  
可行域的确定见下图（当等式约束数量p大于决策变量数量n时，不一定有可行域，因为多条直线必要穿过同一点）：  

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622195445.png)
  

&nbsp;   
## 梯度/海森矩阵
在单变量函数中，梯度是函数的微分，代表着函数在某个给定点的切线斜率。  
在多变量函数中，梯度是空间内的一个向量，梯度的方向是函数在给定点局部上升或下降最快的方向，相当于一元函数的一次导 数。   
海森矩阵（Hessian Matrix）
海森矩阵相当于多元函数求二次导数组成的矩阵，由于$\frac{\partial ^2f}{\partial x_1\partial x_2}$=$\frac{\partial ^2f}{\partial x_2\partial x_1}$，所以海森矩阵是对称矩阵，计算时只需算下三角或上三角。  
n元函数的梯度、海森矩阵的定义及计算示例见下图：  

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622202854.png)

&nbsp; 
## 泰勒展开
泰勒展示式在函数某一点附近用简单多项式近似复杂函数，带来了分析计算上的便利。  
一元函数在$x_0$处的的泰勒式展开为：  
$$f\left( x \right) =f\left( x_0 \right) +f'\left( x_0 \right) \left( x-x_0 \right) +\frac{1}{2!}f''\left( x_0 \right) \left( x-x_0 \right) ^2...+\frac{1}{n!}f^n\left( x_0 \right) \left( x-x_0 \right) ^n+C$$
多元函数在$\boldsymbol{x}_0$处的泰勒级数展开（只取前两项）为：  
$$f\left( \boldsymbol{x} \right) \approx f\left( \boldsymbol{x}_0 \right) +\nabla f\left( \boldsymbol{x} \right) ^{\mathrm{T}}\left( \boldsymbol{x}-\boldsymbol{x}_0 \right) \nabla f\left( \boldsymbol{x} \right) +\frac{1}{2}\left( \boldsymbol{x}-\boldsymbol{x}_0 \right) ^{\mathrm{T}}H\left( \boldsymbol{x} \right) \left( \boldsymbol{x}-\boldsymbol{x}_0 \right)$$
其中$H\left( \boldsymbol{x} \right)$为对于x0处的海森矩阵。  
 
&nbsp; 
## 正定、半正定、负定矩阵   
在线性代数中，正定矩阵（positive definite matrix）对应复数中的正实数。半正定矩阵（positive semi-define matrix）对应复数中的非负实数，负定矩阵（negative definite matrix）对于复数中的负数。  
**正定矩阵的定义：** 若 n 阶实对称方阵 A ，对于任意的 n 维非零向量 x , 满足 $\boldsymbol{x}^{\mathrm{T}}\boldsymbol{Ax}>0$ ，则方阵 A 为正定矩阵。  
**半正定矩阵的定义：** 若 n 阶实对称方阵 A ，对于任意的 n 维向量 x , 满足 $\boldsymbol{x}^{\mathrm{T}}\boldsymbol{Ax}\geqslant 0$ ，则方阵 A 为半正定矩阵。  
**正定矩阵的判断：** n 阶方阵 A 的各阶顺序主子式（左上角K阶方阵构成的行列式）均大于0。  
**负定矩阵的判断：** n 阶方阵 A 的各阶顺序主子式（左上角K阶方阵构成的行列式）负正相间（奇数阶小于0，偶数阶大于0）。
不定矩阵的判断：非正定非负定即为不定矩阵。
协方差矩阵要求是半正定的。     

&nbsp; 
## 凸集/凸函数/凸规划   
一元凸函数定义：若函数 f(x) 曲线上任意两点的连线用在 f(x) 曲线上方，则 f(x) 为**凸函数** 。  
在几何中，存在关系$f\left( \boldsymbol{x} \right) \leqslant Y$  

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622224755.png)

&nbsp;  
凸集定义：集合D中任意两点$\boldsymbol{x}_1$、$\boldsymbol{x}_2$的连线仍在D内，则D内凸集，否则为非凸集。  
多元凸函数定义：f(x)定义域为凸集，对任意实数$0\leqslant \lambda \leqslant 1$，，D中任意两点$\boldsymbol{x}_{\boldsymbol{1}}\text{、}\boldsymbol{x}_2$均存在：
$${\ f\left[ \lambda \boldsymbol{x}_{\boldsymbol{1}}+\left( 1-\lambda \right) \boldsymbol{x}_{\boldsymbol{2}} \right] \leqslant \lambda f\left( \boldsymbol{x}_{\boldsymbol{1}} \right) +\left( 1-\lambda \right) f\left( \boldsymbol{x}_{\boldsymbol{2}} \right) \,\,\ \   \      }0\leqslant \lambda \leqslant 1$$

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622225639.png)
多元凸函数的判定条件为：$f\left( \boldsymbol{x} \right)$的海森矩阵为半正定矩阵。

&nbsp;   
凸规划定义:对于优化问题
$$\underset{\boldsymbol{x}\in R^n}{\min}\,\,f\left( \boldsymbol{x} \right)$$
$$s.t.    g_u\left( \boldsymbol{x} \right) \leqslant 0 ,u=1,2,...m$$
当f(x),g(x)均为凸函数时，优化问题为凸规划问题。  
凸规划问题的性质：

* f(x)等值线呈大圈套小圈的形式
* 可行域为凸集
* 局部最小点一定是全局最优解　　

&nbsp;   
## 算法收敛性
对于式
$$\underset{k\rightarrow \infty}{\lim}\frac{||\boldsymbol{x}_{k+1}-\boldsymbol{x}^*||}{||\boldsymbol{x}_k-\boldsymbol{x}^*||^{\beta}}=\sigma \,\,   \left( 0<\sigma <1 \right)$$
* $\beta =1$时，算法具有线性收敛性/线性收敛速度
* $\beta =2$时，算法具有二次收敛性/二阶收敛速度
* $1<\beta <2$时，算法具有超线性收敛性

&nbsp; 
# 2 极值条件
## 2.1 无约束条件的极值条件  
在一元函数中，通过令$f'\left( x \right) =0$得到极值点x0，再通过考察$f''\left( x_0 \right)$的正负性判断其为极大点、极小点或拐点。  
类似的，在多元函数中，令$f\left( \boldsymbol{x} \right)$的一阶导数$\nabla f\left( \boldsymbol{x} \right)=0$解出极值点$\boldsymbol{x}_0$，再通过考察$f\left( \boldsymbol{x} \right)$二阶导数$H\left( \boldsymbol{x} \right)$是否为正定判断其为极大点、极小点或鞍点（如x1上极大，x2上极小）。    
一元函数及多元函数的极值条件对比表如下图：  

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622214023.png)

&nbsp;   
多元函数极值求解示例如下图：   

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622221846.png)


   


　　　　　　　　　　　

&nbsp;   
## 2.2 约束优化问题的极值条件　　　　

库恩塔克条件(Kuhn-Tucker conditions，KT条件)是确定某点为极值点的必要条件。如果所讨论的规划是凸规划，那么库恩-塔克条件也是充分条件。  
并非所有约束条件都起作用，即并非落在所有约束条件的边界交集处。对于起作用的边界条件，满足边界函数g(x)或h(x)为0.

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210622234120.png)

其中$\,\,\lambda >0$为拉格朗日乘子，当某一点x使$\,\,\lambda >0$存在时，称满足KT条件，则该点x是极值点。  
约束优化问题的应用示例如下图：  

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210623103031.png)






&nbsp;   
&nbsp;   
# 3 优化设计方法及迭代
优化设计方法的基本思路是搜索、迭代、逼近。  
基本迭代公式为：
$$\boldsymbol{x}_{k+1}=\boldsymbol{x}_k+\alpha _k\boldsymbol{s}_k$$
其中$\alpha _k$为第k步迭代步长，$\boldsymbol{s}_k$为第k步迭代方向。  
迭代的终止条件类型主要有三种（$\varepsilon$为终止参数，需设定）：
* 点距：$\left\| \boldsymbol{x}_{k+1}-\boldsymbol{x}_k \right\| <\varepsilon$
* 值差：$\left\| f\left( \boldsymbol{x}_{k+1} \right) -f\left( \boldsymbol{x}_k \right) \right\| <\varepsilon$
* 梯度：$\left\| \nabla f\left( \boldsymbol{x}_{k+1} \right) \right\| <\varepsilon$

&nbsp;
## 3.1 一维搜索方法
一维搜索是指通过迭代逼近单变量函数的最优值，多维目标函数的极值点优化问题需要一系列一维搜索方法。  
一维搜索方法的主要步骤为
* 1.确定搜索区间（单峰区间）
* 2.找到最优步长，缩小单峰区间，找到满足精度要求的最优值x*  

单峰区间是指一个高低高的区间，其中必存在一个极值。  
缩小单峰区间的方法主要有
* 0.618法（黄金分割法）
* 二次插值法
* 三次插值法

### 寻找单峰区间
单峰区间的寻找首先输入函数，初始步长，初始坐标，通过前进寻找到函数值为高低高的区间[a,b]。  
单峰区间的确定方法如下图：

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210623160224.png)


### 缩小单峰区间
单峰区间找到后，需要一步步缩小单峰区间直到找到满足精度的 x*

1.随机删除法（通用思想）
在a,b区间随机选取x1 < x2,按照如下原则删除冗余区间，获得更小的高低高区间，直到达到精度。   
随机法的逼近原则如下： 

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210623155554.png)
 
2.黄金分割法
黄金分割法首先选取在区间中几何对称（0.618倍的区间长度）的两个点x1,x2，比较两点函数值后缩小单峰区间。  
黄金分割法的逼近步骤如下图：  

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210623160138.png)

黄金分割法的特点主要有：
* 缩短率相等，均为0.618
* 缩短率均为0.618时，每次迭代只算一个新的点（x1或x2）函数值，另一点不变。方便计算
* 对函数可微等性质无要求，只需要代点求解出函数值
* 未使用函数的性态，收敛速度稳定但是慢。


&nbsp;   
## 3.2 无约束优化方法  
包括直接法：只使用函数目标值，不必求导（坐标轮回，Powell,单纯形法）；间接法/解析法：利用函数值及函数梯度、海森矩阵（梯度法、共轭梯度法、牛顿法、DFP变尺度法） 
基本问题:
$$\min  f\left( \boldsymbol{x} \right) =\frac{1}{2}\boldsymbol{x}^TA\boldsymbol{x}+B^T\boldsymbol{x}+C$$

&nbsp; 
### **梯度法/最速下降法**
多元函数的梯度是函数值变化最快的方向，基于此，梯度法将迭代的方向$s_k$设定为负梯度方向，迭代的步长通过一维搜索方法来寻找最优化的步长$\alpha _k$
梯度法的迭代步骤及示例如下图所示：

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210623233915.png)

梯度法在极值点的外围搜索速度快，在内圈搜索速度慢。


&nbsp; 
### **共轭梯度法**  
由于原生梯度法在极值点的外围搜索速度快，在内圈搜索速度慢。而共轭法具有二次收敛性，在内圈有较好的收敛性能，因此共轭梯度法将二者融合。共轭梯度法在开始时使用梯度法收敛，从第二次开始搜索方向根据共轭条件对负梯度进行修正，沿着修正后的共轭方向迭代逼近极值点x*，其中$\beta ^k=\frac{\left\| g_{k+1} \right\| ^2}{\left\| g_k \right\| ^2}$为共轭系数，$\beta ^k=\frac{\left\| g_{k+1} \right\| ^2}{\left\| g_k \right\| ^2}$使$s_k$和$s_{k+1}$关于A矩阵共轭。  
共轭梯度优化过程如下图：

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210623233758.png)


&nbsp; 
### **牛顿法**  
牛顿法在$x_k$点处做二阶泰勒展开，在$x_k$邻域内用展开后的二次函数$\varphi \left( x \right)$近似f(x),然后求出$\varphi \left( x \right)$的极值点$x_{\varphi}^{*}$,当f(x)为二次函数时，认为最优值即为$x_{\varphi}^{*}$，否则继续迭代。
原始牛顿法的步长$\alpha _k$恒定为1，搜索效率低，容易越过极值点。改进后的牛顿法（修正/广义/阻尼牛顿法）在迭代时，通过一维搜索选取最优的迭代步长$\alpha _k$,不再机械地取$\alpha _k$为1。  
通过与基本迭代公式相比较：
$$\text{基本迭代公式：}\boldsymbol{x}_{k+1}=\boldsymbol{x}_k+\alpha _k\boldsymbol{s}_k$$
$$\text{一维牛顿方向为：}\boldsymbol{s}_k=-\frac{f'\left( \boldsymbol{x}_k \right)}{f''\left( \boldsymbol{x}_{k+1} \right)}$$
$$\text{高维牛顿方向为：}\boldsymbol{s}_k=-H\left( \boldsymbol{x}_k \right) ^{-1}\nabla f\left( \boldsymbol{x}_k \right)$$

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210624144702.png)

牛顿法的特点：
* 具有二阶收敛性能
* 需要计算海森矩阵逆矩阵，高维计算量大


&nbsp;    
### **DFP变尺度法/拟牛顿法**
DFP法基于梯度法和牛顿法提出，既有二阶收敛性，又可以不必求海森矩阵，海森矩阵逆矩阵，应用范围广。  
DFP法的思想是用 n 阶方阵$A_k$,在计算中递推逼近$H\left( \boldsymbol{x}_k \right) ^{-1}$。  
若想要用 n 阶方阵$A_k$逼近$H\left( \boldsymbol{x}_k \right) ^{-1}$ ，需要满足变尺度条件：  
$$A_{k+1}\bigtriangleup g_k\,\,=\bigtriangleup \boldsymbol{x}_k\,\, \text{，其中}\bigtriangleup g_k=\nabla f\left( \boldsymbol{x}_k \right)$$
DFP的迭代公式为：
$$\boldsymbol{x}_{k+1}=\boldsymbol{x}_k-\alpha _kA_k\nabla f\left( \boldsymbol{x}_k \right)$$
特殊地，当$A_k=I$时，迭代公式为梯度法迭代公式。  
当$A_k=H\left( \boldsymbol{x}_k \right) ^{-1}$时，迭代公式为牛顿法的迭代公式。  
在第一轮学习中，选取 $A_0=I$，第一轮使用梯度法迭代，在第k+1轮时，不断校正 $A_k$ 使得 $A_k$ 更加逼近 $H\left( \boldsymbol{x}_k \right) ^{-1}$,使用以下公式进行迭代：
$$A_{k+1}=A_k+E_k$$
$$DFP\text{法中校正矩阵}  E_k=\frac{\bigtriangleup \boldsymbol{x}_k\left( \bigtriangleup \boldsymbol{x}_k \right) ^T}{\left( \bigtriangleup \boldsymbol{x}_k \right) ^T\bigtriangleup g_k}-\frac{A_k\bigtriangleup g_k\left( \bigtriangleup g_k \right) ^T\left( A_k \right) ^T}{\left( \bigtriangleup g_k \right) ^TA_k\bigtriangleup g_k}$$
由于DFP中校正矩阵将$A_k$放置于分母中，舍入误差和一维搜索不精确坑导致某变尺度矩阵奇异，BFGS法对此做出改善，将$A_k$置于分子，校正矩阵的计算公式为：
$$BFGS\text{法中校正矩阵} E_k=\frac{1}{\left( \bigtriangleup \boldsymbol{x}_k \right) ^T\bigtriangleup g_k}\left\{ \bigtriangleup \boldsymbol{x}_k\left( \bigtriangleup \boldsymbol{x}_k \right) ^T+\frac{\bigtriangleup \boldsymbol{x}_k\left( \bigtriangleup \boldsymbol{x}_k \right) ^T\left( \bigtriangleup g_k \right) ^TA_k\bigtriangleup g_k}{\left( \bigtriangleup \boldsymbol{x}_k \right) ^T\bigtriangleup g_k}-A_k\bigtriangleup g_k\left( \bigtriangleup \boldsymbol{x}_k \right) ^T-\bigtriangleup \boldsymbol{x}_k\left( \bigtriangleup g_k \right) ^TA_k \right\}$$

DFP法的迭代框图如下图：

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210624145241.png)

DFP法的特点：
* 二阶收敛
* 每次搜索产生的方向都是共轭的
* 递推的公式便于迭代
* 计算方便，无需计算海森矩阵和其逆矩阵

&nbsp;   
无约束优化方法的特点如下图：

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210626154646.png)

&nbsp;   
&nbsp;   
## 3.3 约束优化方法
数学模型：
$$\underset{\boldsymbol{x}\in R^n}{\min}\,\,f\left( \boldsymbol{x} \right)$$
$$s.t. \  \ \    g_u\left( \boldsymbol{x} \right) \leqslant 0 ,u=1,2,...m$$
$$h_v\left( \boldsymbol{x} \right) =0,v=1,2,...p\left( p<n \right) \,\,$$
主要分为直接法：在可行域内直接探索最优解（随机方向法，复合形法，简约梯度法等）；间接法：将约束优化问题转化为无约束约束优化问题，用无约束优化方法进行求解（惩罚函数法，广义乘子法等）

### **复合形法**
是无约束优化中单纯形法的多维推广。  
主要思路：  
* 在可行域内产生 k (k>n+1)个顶点，构成初始复合形（多面体）
* 舍弃最差的点（函数值最大）
* 用可以使目标函数有所下降又满足约束条件的一个新点代替被舍弃的最差的点
* 迭代直到满足终止条件

初始顶点的产生主要有两种方法：1.全部人工选取可行域内的点 2.人工选取一个可行域内的点后其他由随机产生。
复合形法常用的反射系数为1.3，优化步骤如下图：

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/fuhexing.png)

复合形法的特点：
* 不计算导数，无特殊要求
* 无需一维搜索，只需比较函数
* 维数大于5时，收敛慢


&nbsp;   
### **惩罚函数法**
根据约束函数的特点，构造惩罚项到目标函数中，将约束优化变成无约束优化问题。  


&nbsp;  
### **内惩罚函数法/内点法**
内点法的通过构造惩罚项，使得迭代点都在可行域内,当迭代的点靠近边界时，通过惩罚项将该点拉回可行域。
约束优化问题：
$$\underset{\boldsymbol{x}}{\min}\,\,f\left( \boldsymbol{x} \right)$$
$$s.t.    g_u\left( \boldsymbol{x} \right) \leqslant 0 ,u=1,2,...m$$
构造的内罚函数为
$$\varphi \left( \boldsymbol{x},r_k \right) =\,\,f\left( \boldsymbol{x} \right) -r_k\sum_{u=1}^m{\frac{1}{g_u\left( \boldsymbol{x} \right)}}$$
其中$r_k$为内罚因子（$r_0$可取1~50），是递减的正数序列，$-r_k\sum_{u=1}^m{\frac{1}{g_u\left( \boldsymbol{x} \right)}}$为惩罚项。
$$r_0>r_1...>r_k...>0$$

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210625134035.png) 

内点法初始点的选取:
* 1.原设计方案（原初始点）  
* 2.随机产生  
* 3.搜索法。先产生一个只满足部分约束的点，再迭代使得该点满足所有约束，如下图：

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210625140252.png)


内点法特点：
* 内罚函数在可行域内进行，得到的解总是可行的
* 只能求解不等式约束优化问题，不能解等式约束优化问题
* 初始点必须要在可行域内



&nbsp; 
### **外惩罚函数法/外点法**  
外点法是一种从随机一点（一般在可行域外）出发，逐渐移动到可行区域的方法。
对于约束优化问题：  
$$\underset{\boldsymbol{x}\in R^n}{\min}\,\,f\left( \boldsymbol{x} \right)$$
$$s.t.    g_u\left( \boldsymbol{x} \right) \leqslant 0 ,u=1,2,...m$$
$$h_v\left( \boldsymbol{x} \right) =0,v=1,2,...p\left( p<n \right) \,\,$$
构造外惩罚函数：
$$\varphi \left( \boldsymbol{x},M_k \right) =f\left( \boldsymbol{x} \right) +M_k\left\{ \sum_{u=1}^m{\left( \max \left[ g_u\left( \boldsymbol{x} \right) ,0 \right] \right) ^2+\sum_{v=1}^p{\left( h_v\left( \boldsymbol{x} \right) \right) ^2}} \right\}$$
其中$M_k$为外罚因子，$M_0<M_1<M_2...<M_k...\infty$为递增的正数序列，一般M0=1，递增系数c=5~8。  
当x满足所有的约束条件时，惩罚项为0；当x不满足约束条件（在可行域外时），惩罚项起作用，且离约束边界越远，惩罚项的值越大，惩罚作用越明显。  
终止准则（收敛准则）：
* 1.$Q\leqslant \varepsilon _1\,\,=10^{-3}~10^{-4}$，Q为当前最优点代入多个约束函数中的最大值。
* 2.$M>R\text{且}\left\| x*\left( M_{k+1} \right) -x*\left( M_k \right) \right\| \leqslant \varepsilon _2$，R为外罚因子控制量。   

外点法的迭代图如下：

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210625165916.png)


外点法的示例如下图：

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210625180418.png)



外罚函数的特点：
* 对初始点无要求，可在可行域外
* 可解不等式优化和等式优化
* 最后的解是可行域附近（外）的不可行解，只能近似。对此可设置约束裕度（$\varDelta>0$）,将边界缩小（$g_u\left( \boldsymbol{x} \right) +\varDelta \leqslant 0$），使得一句缩小后边界解出的点在原边界内。


&nbsp; 
### **混合罚函数法**
混合罚函数法将内罚和外罚函数法进行结合，使得初始点不必在可行域内，且可以解等式和不等式问题。  
混合罚函数法的一般形式为：  
$$\varphi \left( \boldsymbol{x},M_k \right) =f\left( \boldsymbol{x} \right) -r_k\sum_{u=1}^m{\frac{1}{g_u\left( \boldsymbol{x} \right)}+\frac{1}{\sqrt{r_k}}}\sum_{v=1}^p{\left( h_v\left( \boldsymbol{x} \right) \right) ^2}$$
其中罚因子$r_0>r_1>r_2...>r_k...>0$，逐渐趋于0.混合罚函数的迭代方法与外罚函数的迭代方法类似。

&nbsp;  

&nbsp;  
### **拉格朗日乘子法与KKT条件**

### **拉格朗日乘子法**    
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
### **KKT条件** 

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




&nbsp; 
# 4 遗传算法
遗传算法基于适者生存，不适者淘汰的原理，对问题进行求解。遗传算法主要包括**选择**（将适应度更高的值赋予更大的选中概率），**交叉**（模拟生物杂交，将两个母体位于交叉位置后的字符串互换）和**变异**（模拟基因突变，以一定概率从种群中选取某个个体的某个基因进行翻转）   

下图为遗传算法执行步骤：

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210626093754.png)
  

下图为遗传算法的示例：  

![](https://cdn.jsdelivr.net/gh/Townjj/Markdown-Images/Machine-Learning/20210626100325.png)


遗传算法具有以下特点：
* 遗传算法具有鲁棒性(稳定的收敛性)。
* 对问题变量的编码集操作，具有适应性。
* 从一组初始点出发，具有并行性，避免陷入局部最小点。
* 仅使用目标函数，不用求导
* 启发式搜索，不是简单的枚举，计算量与解题规模呈线性增长，避免了维数灾难。 