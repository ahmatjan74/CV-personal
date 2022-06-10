import torch
import torch.nn as nn
import torchvision
import os
from vggnet import VGGNet
from resnet import resnet
from mobilenetv1 import mobilenetv1_small
from loadcifar10_2 import train_data_loader, test_data_loader
import tensorboardX


if __name__ == '__main__':
    device = torch.device("cude" if torch.cuda.is_available() else "cpu")

    epoch_num = 200
    lr = 0.01
    batch_size = 128

    model_path = "models"
    log_path = "logs/pytorch_vgg"
    writer = tensorboardX.SummaryWriter(log_path)

    # net
    net = VGGNet()
    # net = resnet()
    net = mobilenetv1_small()

    # loss
    loss_func = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=5,
                                                gamma=0.9)
    step_n = 0
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

            writer.add_scalar("train loss", loss.item(), global_step=step_n)
            writer.add_scalar("train correct",
                              100.0 * correct.item() / batch_size, global_step=step_n)

            im = torchvision.utils.make_grid(inputs)
            writer.add_image("train im", im, global_step=step_n)

            step_n += 1

            # print("step", i, "loss is", loss.item(), 'mini-batch correct is:',
            #       100.0 * correct / batch_size)

        torch.save(net.state_dict(), "{}/{}.pth".format(model_path,
                                                        epoch + 1))
        scheduler.step()

        print("lr:", optimizer.state_dict()["param_groups"][0]["lr"])

        # test
        sum_loss = 0
        sum_correct = 0
        for i, data in enumerate(test_data_loader):
            net.eval()
            inputs, labels = data
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            _, pred = torch.max(outputs.data, dim=1)
            correct = pred.eq(labels.data).sum()

            sum_loss += loss.item()
            sum_correct += correct.item()
            im = torchvision.utils.make_grid(inputs)
            writer.add_image("test im", im, global_step=step_n)

        test_loss = sum_loss * 1.0 / len(test_data_loader)
        test_correct = sum_correct * 100.0 / len(test_data_loader) / batch_size

        writer.add_scalar("test loss", test_loss, global_step=epoch + 1)
        writer.add_scalar("test correct",
                          test_correct, global_step=epoch + 1)

        print("epoch is", epoch + 1, "loss is:", test_loss,
              "test correct is:", test_correct)
