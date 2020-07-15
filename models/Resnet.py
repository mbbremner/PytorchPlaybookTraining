'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import KLDivLoss
import numpy as np
from torch.distributions import kl_divergence
from torchsummary import summary

def compute_KL_loss(low, med, high, T):
    """
    Compute the KL divergence loss from 3 layers
    :param low: lowest level feature layer
    :param med: middle feature layer
    :param high: highest level feature layer
    :param T: temperature softening parameter
    :return: KL_loss
    """
    low, med, high = F.softmax(low, dim=0), F.softmax(med, dim=0), F.softmax(high, dim=0)
    low = apply_clsa_softening(low, T).log()
    med = apply_clsa_softening(med, T).log()

    return (T ** 2) * (KLDivLoss(reduction='batchmean')(med, high)
                       + KLDivLoss(reduction='batchmean')(low, high))


class CLSA_Block(nn.Module):

    def __init__(self, block_expansion):
        super(CLSA_Block, self).__init__()


        fc_features_in = block_expansion * 8192
        # self.bn =  nn.ReLU()
        self.clsa_block = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),
                                        nn.Flatten(),
                                        nn.Linear(fc_features_in, 256),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.ap1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flat1 = nn.Flatten()
        self.fc1 = nn.Linear(fc_features_in, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()


        # input = input.view(input.size(0), -1)

    def forward(self, x):
        out = self.ap1(x)
        out = self.flat1(out)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        self.loss_function = torch.nn.CrossEntropyLoss()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        out = F.avg_pool2d(out5, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_CLSA(ResNet):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_CLSA, self).__init__(block, num_blocks, num_classes)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.KL_loss_function = torch.distributions.kl.kl_divergence

        self.clsa1 = CLSA_Block(block_expansion=4)
        self.clsa2 = CLSA_Block(block_expansion=2)
        self.clsa3 = CLSA_Block(block_expansion=1)

        self.linear = nn.Linear(256, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)


        clsa_out_1 = self.clsa1(out3)
        clsa_out_2 = self.clsa2(out4)
        clsa_out_3 = self.clsa3(out5)

        prediction = self.linear(clsa_out_3)

        return (clsa_out_1, clsa_out_2, clsa_out_3, prediction)



def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet50_CLSA():
    return ResNet_CLSA(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():

    net = ResNet50()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    print(y)
    m = torch.nn.Softmax(dim=1)
    print(m(y))


def test_clsa():

    net = ResNet50_CLSA()
    y = net(torch.randn(1, 3, 32, 32))
    print([item.size() for item in y])
    # print(y)
    # m = torch.nn.Softmax(dim=1)
    # print(m(y))

def Test(net):
    # prediction = net(torch.randn(2, 3, 32, 32))
    # loss = torch.nn.CrossEntropyLoss()
    # labels = torch.tensor([0, 1])
    # print(loss(prediction, labels))

    # CLSA
    Temp = 0.1
    clsa1, clsa2, clsa3, prediction = net(torch.rand(16, 3, 32, 32))


    CE_loss = torch.nn.CrossEntropyLoss()(F.softmax(prediction, dim=0), torch.tensor([1]*16))
    KL_loss = compute_KL_loss(clsa1, clsa2, clsa3, Temp)
    loss = CE_loss + KL_loss
    loss.backward()

    print("CE_loss: ", CE_loss)
    print("KL_loss: ", KL_loss)
    print("Total_loss: ", loss)
    summary(net.cuda(), input_size=(3,32,32))
    exit()

    total_loss = CE_loss + KL_loss
    # kl_loss = torch.distributions.kl_divergence(F.log_softmax(clsa1), F.softmax(clsa3))
    print(CE_loss, KL_loss, total_loss)
    exit()
    # print(kl_loss)
    # print([item.size() for item in y])

def apply_clsa_softening(semantic_vector, temp):

    print(temp)
    soft_top = [np.exp((1 / temp) * item) for i, item in enumerate(semantic_vector.detach().numpy())]
    soft_bottom = [sum(np.exp((1 / temp) * item.detach().numpy())) for item in semantic_vector]

    return torch.tensor([item / soft_bottom[i] for i, item in enumerate(soft_top)])

if __name__ == "__main__":
    net = ResNet50_CLSA()
    net.eval()
    Test(net)
