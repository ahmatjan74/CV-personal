import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1 * 28 * 28  => maxPooling 2 下采样 => 14 * 14
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc = torch.nn.Linear(14 * 14 * 32, 10)

    def forward(self, x):
        out = self.conv(x)
        # 拉成一个向量
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

# class CNN(torch.nn.Module):
#     def __int__(self):
#         super(CNN, self).__init__()
#         self.conv1 = torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
#             torch.nn.BatchNorm2d(32),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2)
#         )
#         # 1 * 28 * 28  => maxPooling 2 下采样 => 14 * 14
#         self.fc = torch.nn.Linear(in_features=14 * 14 * 32, out_features=10)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = out.view(out.size()[0], -1)
#         out = self.fc(out)
#         return out
