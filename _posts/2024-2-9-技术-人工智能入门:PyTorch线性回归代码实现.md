```
import torch

import matplotlib.pyplot as plt
# 一元线性回归代码的模拟实现
torch.manual_seed(10)
lr = 0.05 
# 设置随机分布种子
# 设置学习率

x = torch.rand(20,1) * 10 # x data (tensor), shape=(20, 1)
# 生产20个 1～10的小数（float）
y = 2*x + (5 + torch.randn(20, 1)) # y data (tensor), shape=(20, 1)
# 根据预先设置好的线性方程，计算标签。其中randn设置随机偏移量，让数据更加真实。


w = torch.randn((1), requires_grad=True)
# w的默认权重
b = torch.zeros((1), requires_grad=True)
# b的默认权重，requires_grad 表示，这几个张量需要自动求导，会自动保存运算路线，便于反向传播


for iteration in range(1000):
# 最多1000次迭代


# # 前向传播

wx = torch.mul(w, x)
# 计算 w * x
y_pred = torch.add(wx, b)

# 计算预测值，w * x + b


# # 计算 MSE loss

loss = (0.5 * (y - y_pred) ** 2).mean()
# 均方损失


# # 反向传播

loss.backward()

# loss为output，对各个叶子结点求导

# # 更新参数

torch.sub(b, lr * b.grad)

w.data.sub_(lr * w.grad)

# 更新 b 和 w，b = b - lr * grad(b) , w = w - lr * grad(w)
# sub_ 是自替换函数，改变叶子结点的值

# # 清零张量的梯度 20191015增加

w.grad.zero_()

b.grad.zero_()
# 清除梯度，（将计算路线图从内存中删除，防止下一次的梯度叠加）

# # 绘图

if iteration % 20 == 0:

plt.cla() # 防止社区版可视化时模型重叠2020-12-15

plt.scatter(x.data.numpy(), y.data.numpy())

plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)

plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})

plt.xlim(1.5, 10)

plt.ylim(8, 28)

plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))

plt.pause(0.5)



if loss.data.numpy() < 1:

break

plt.show()
```