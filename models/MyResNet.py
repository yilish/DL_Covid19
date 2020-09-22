import torch.nn as nn
import torch

class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return nn.functional.avg_pool2d(x, kernel_size=x.size()[2:])


# 残差模块


class Residual(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=False):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=stride)
        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(False)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        if self.downsample:
            residual = self.downsample(residual)
            x += residual
        return self.relu(x)


def resnet_layer(in_channel, out_channel, block_num, stride=1):
    layer = []
    for i in range(block_num):
        if i == 0:
            layer.append(Residual(in_channel, out_channel, stride, False))
        else:
            layer.append(Residual(out_channel, out_channel, stride, True))
    return nn.Sequential(*layer)


class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # self.net.add_module('layer1:', resnet_layer(64, 64, 2, 2))
        # self.net.add_module('layer2:', resnet_layer(64, 128, 2, 2))
        # self.net.add_module('layer3:', resnet_layer(128, 256, 2, 2))
        # self.net.add_module('layer4:', resnet_layer(256, 512, 2, 2))
        # self.net.add_module('global_pool:', GlobalAvgPool2d())
        # self.net.add_module('fc_layer:', nn.Linear(512, 10, True))

        self.layer1=resnet_layer(64, 64, 2, 2)
        self.layer2=resnet_layer(64, 128, 2, 2)
        self.layer3=resnet_layer(128, 256, 2, 2)
        self.layer4=resnet_layer(256, 512, 2, 2)
        self.global_pool=GlobalAvgPool2d()
        self.fc=nn.Linear(512, 10, True)
        torch.autograd.set_detect_anomaly(True)

    def forward(self, x):
        x=self.net(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.global_pool(x)
        x=x.view(x.size(0), -1)
        x=self.fc(x)
        return x