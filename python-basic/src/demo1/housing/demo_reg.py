import torch
import numpy as np
import re  # regular expression
from net import Net

# data
ff = open("housing.data").readlines()
data = []
for item in ff:
    out = re.sub(r"\s{2,}", " ", item).strip()
    data.append(out.split(" "))

data = np.array(data).astype(float)
print(data.shape)

Y = data[:, -1]
X = data[:, 0: -1]

Y_train = Y[0:496, ...]
Y_test = Y[496:, ...]
X_train = X[0:496, ...]
X_test = X[496:, ...]

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# net
net = Net(13, 1)

# loss
loss_fun = torch.nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# training
for i in range(10000):
    x_data = torch.tensor(X_train, dtype=torch.float32)
    y_data = torch.tensor(Y_train, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss = loss_fun(pred, y_data) * 0.001

    # 梯度为0
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    print("ite: {}, loss: {}".format(i, loss))
    print(pred[0:10])
    print(y_data[0:10])

    # test
    x_test_data = torch.tensor(X_test, dtype=torch.float32)
    y_test_data = torch.tensor(Y_test, dtype=torch.float32)
    pred = net.forward(x_test_data)
    pred = torch.squeeze(pred)
    loss_test = loss_fun(pred, y_test_data) * 0.001
    print("ite:{}, loss_test:{}".format(i, loss_test))

torch.save(net, "model/model.pkl")
# torch.load("")
# torch.save(net.state_dict(), "param.pkl")
# net.load_state_dict("")