import torch
import numpy as np
import cv2

a = torch.rand(2, 2) * 10
print(a)

a = a.clamp(2, 5)  # 2< a < 5
print(a)

b = torch.tensor([[2, 3], [4, 5]], dtype=torch.float32)
print(b)

a = torch.rand(4, 4)
b = torch.rand(4, 4)
print(a)
print(b)
out = torch.where(a > 0.5, a, b)
print(out)

a = torch.linspace(1, 16, steps=16).view(4, 4)
print(a)

mask = torch.ge(a, 8)
out = torch.masked_select(a, mask)
print(out)

a = torch.zeros(2, 4)
b = torch.ones(2, 4)
print(a)
print(b)
out = torch.cat((a, b), dim=0)
print(out)
out = torch.cat((a, b), dim=1)
print(out)

a = torch.rand(3, 4)
out = torch.chunk(a, 2, dim=1)
print(a)
print(out[0])
print(out[1])

a = torch.rand(10, 4)
out = torch.split(a, 3, dim=0) # same  as chunk
print(len(out))
for i in range(len(out)):
    print(out[i])

out = torch.split(a, [1, 3, 6], dim=0)
for i in range(len(out)):
    print(out[i])

a = torch.rand(2, 3)
print(a)
out = torch.reshape(a, (1, -1))
print(out)

out = torch.reshape(a, (3, 2))
print(out)

a = torch.rand(2, 3)
b = torch.full((2, 3), 3)
print(a)
print(b)


a = np.zeros([2, 3])
out = torch.from_numpy(a)
print(a)
print(out)

data = cv2.imread('test.png')
print(data)

out = torch.from_numpy(data)
print(out)

out = torch.flip(out, dims=[0])
data1 = out.numpy()

# cv2.imshow('test1', data1)
# cv2.waitKey(0)

x = torch.ones(2, 2, requires_grad=True)
x.register_hook(lambda grad: grad * 2)

y = x + 2
z = y * y * 3
nn = torch.ones(2, 2)
torch.autograd.backward(z, grad_tensors=nn, retain_graph=True)
z.backward(nn, retain_graph=True)
print(torch.autograd.grad(z, [x], grad_outputs=nn))
print(x.grad)
print(y.grad)
print(z.grad_fn)


