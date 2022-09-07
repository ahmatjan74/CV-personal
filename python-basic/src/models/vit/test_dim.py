import torch
import os
import numpy as np

# a = torch.randn((3,4))
# print(a)
# print(a.size())
# # dim = 1 ? 行
# b = torch.argmax(a, dim=1) # 将输入input张量，无论有几维，首先将其reshape排列成一个一维向量，然后找出这个一维向量里面最大值的索引
# print(b)
# print(b.size())
# # dim = 0 ？ 列
# c = torch.argmax(a, dim=0) # 将输入input张量，无论有几维，首先将其reshape排列成一个一维向量，然后找出这个一维向量里面最大值的索引
# print(c)
# print(c.size())

# 一维向量
# t1 = torch.tensor((1, 2))
# # 二维向量
# t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
# # 三维向量
# t3 = torch.tensor([[[1, 2], [3, 4]],[[5, 6], [7, 8]]])
# print(t1.ndim, t2.ndim, t3.ndim, sep = ', ')
# # 1, 2, 3
# # t1为1维向量
# # t2为2维矩阵
# # t3为3维张量
# print(t1.shape, t2.shape, t3.shape, sep = ', ')
# n = torch.randn(5, 5, 1024)
# print(n.size(), n.size(0), n.size(1), n.size(2))
# a = 13 / 4
# print(a)
# a = 13//4
# print(a)

# from einops import rearrange, repeat
# cls_token = torch.rand(1, 1, 5)
# print(cls_token)
# a = repeat(cls_token, '() n d -> b n d', b=2) 
# print(a)
a = torch.tensor([[2,2], [1, 4]])
b = torch.tensor([[3,5], [1,3]])
c = torch.cat((a, b), dim=1)
d = torch.cat((a, b), dim=0)

print(c)
print(d)