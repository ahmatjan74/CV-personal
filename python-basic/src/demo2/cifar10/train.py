import torch
import torch.nn as nn
import torchvision
import os
from vggnet import VGGNet
from loadcifar10_2 import train_data_loader, test_data_loader


device = torch.device("cude" if torch.cuda.is_available() else "cpu")

epoch_num = 200
lr = 0.01
batch_size = 128

# net
net = VGGNet().to(device)

# loss
loss_func = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=5,
                                            gamma=0.9)

for epoch in range(epoch_num):
    net.train()  # train BN dropout
    print("epoch is: ", epoch)

    for i, data in enumerate(train_data_loader):
        inputs, labels = data

        outputs = net(inputs)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(outputs.data, dim=1)
        correct = pred.eq(labels.data).sum()

        print("step", i, "loss is", loss.item(), 'mini-batch correct is:',
              100.0 * correct / batch_size)

    if not os.path.exists("models"):
        os.mkdir("models")

    torch.save(net.state_dict(), "models")




