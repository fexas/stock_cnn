import torch
import torch.nn as nn

class SA_Net(nn.Module):
    def __init__(self, use_bn=True, dropout_rate=0.5):
        """
        参数：
          use_bn: 是否使用 BatchNorm2d（默认为 True）
          dropout_rate: Dropout 的比率（默认为 0.5）
        """
        super(SA_Net, self).__init__()
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate

        # 构造第一层
        layers1 = [nn.Conv2d(1, 64, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(12, 1))]
        if use_bn:
            layers1.append(nn.BatchNorm2d(64))
        layers1.extend([
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1))
        ])
        self.layer1 = nn.Sequential(*layers1)

        # 构造第二层
        layers2 = [nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(12, 1))]
        if use_bn:
            layers2.append(nn.BatchNorm2d(128))
        layers2.extend([
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1))
        ])
        self.layer2 = nn.Sequential(*layers2)

        # 构造第三层
        layers3 = [nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(12, 1))]
        if use_bn:
            layers3.append(nn.BatchNorm2d(256))
        layers3.extend([
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1))
        ])
        self.layer3 = nn.Sequential(*layers3)

        # 全连接层
        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(46080, 2)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 模型要求输入形状为 [-1, 1, 64, 60]
        x = x.reshape(-1, 1, 64, 60)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(-1, 46080)
        x = self.fc1(x)
        # 若需要 softmax，可取消下行注释
        # x = self.softmax(x)
        return x