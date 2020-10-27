## 线性回归公式

###作者：wangzhong



###模型定义

$$
h_\theta(x^i) = \theta_0 + \theta_1x_1+...+\theta_mx_m
$$

可以简化为：
$$
h_\theta(x^i) = \theta^TX
$$

---

###损失函数

线性回归的损失函数用***均方误差***表示，即最小二乘法
$$
J(\theta)=\frac{1}{2m}\sum^m_{i=0}{(h_\theta(x^i) - y)^2}
$$

---

### 梯度下降

对任意的θ，对其求偏导，通过梯度的值去逼近θ的最优解，使损失函数逼近最小值
$$
\theta_j = \theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)
$$
其中
$$
\begin{aligned}
\frac{\partial}{\partial\theta_j}J(\theta) &= \frac{\partial}{\partial\theta_j}\frac{1}{2m}\sum^m_{i=0}{(h_\theta(x^i) - y)^2}\\
&=\frac{1}{m}\sum^m_{i=0}(h_\theta(x^i) - y)\frac{\partial}{\partial\theta_j}(h_\theta(x^i) - y)\\
&=\frac{1}{m}\sum^m_{i=0}(h_\theta(x^i) - y)x^i_j
\end{aligned}
$$
对于每一个θ，都用梯度下降去逼近最优解，其中自己设定适合的learningrate α和迭代次数