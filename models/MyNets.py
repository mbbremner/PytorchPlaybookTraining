""" Contains a mixture of experimental models (experiment doesn't necessarily mean useful or well-refined)
    and imitations of popular models """

import torch
import torch.nn as nn
import torch.nn.functional as F


# VGG Convolution block
class VggBlock(nn.Module):

    def __init__(self, c_in, c_out, n_layers, p_drop=0):
        """
        :param c_in: # of channels in
        :param c_out: # of of channels out
        :param n_layers: # number of layers for this conv block
        :param p_drop: dropout probability, default 0 skips addition of a dropout
        """
        super(VggBlock, self).__init__()
        self.ceil = False
        self.conv_layer = nn.Sequential()
        for layer in range(n_layers):

            if layer == 0:
                self.conv_layer.add_module('conv' + str(layer), nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1))
            else:
                self.conv_layer.add_module('conv' + str(layer), nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, padding=1))
            self.conv_layer.add_module('bn' + str(layer), nn.BatchNorm2d(c_out))
            self.conv_layer.add_module('relu' + str(layer), nn.ReLU(inplace=True))
            if p_drop != 0:
                self.conv_layer.add_module('dropout' + str(layer), nn.Dropout2d(p=p_drop))
        self.conv_layer.add_module('mp', nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=self.ceil))

    def forward(self, x):

        x = self.conv_layer(x)
        return x


# Simple FC Block with 3 layers + len(n_classes_ layer
class FcBlock(nn.Module):

    def __init__(self, outputs, sizes):
        super(FcBlock, self).__init__()
        # self.x1 = self.x_val
        # self.x2 = int(self.x_val / 2)
        # self.x3 = int(self.x_val / 4)
        # self.num_outputs = outputs
        self.fc_layer = nn.Sequential(
            # Linear block 1
            nn.Linear(sizes[0], sizes[0]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(sizes[0]),
            nn.Dropout(p=0.1),

            # Linear block 2
            nn.Linear(sizes[0], sizes[1]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(sizes[1]),
            nn.Dropout(p=0.1),

            # Linear block 3
            nn.Linear(sizes[1], sizes[2]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(sizes[2]),
            nn.Dropout(p=0.1),

            nn.Linear(sizes[2], outputs),
        )

    def forward(self, x):
        x = self.fc_layer(x)
        return x


# Generalized VGG Class
class VGG(nn.Module):

    def __init__(self, num_outputs, input_shape=(32, 32), **kwargs):
        """CNN Builder."""
        super(VGG, self).__init__()

        self.loss_function = torch.nn.CrossEntropyLoss()

        # Defaults
        self.ceil = False
        self.name = 'vgg16'
        self.default_dropout_vals = (0.3, 0.3, 0.4, 0.4, 0.5)

        # Compute FC Block inputs size:
        self.fc_input_size = int((input_shape[0]/32) * (input_shape[1]/32)) * 512
        # print("FC input size is %d" % self.fc_input_size)
        self.fc_sizes = (self.fc_input_size, 256, 128)
        self.num_outputs = num_outputs

        # Initialize block elements
        self.convBlock = nn.Sequential()
        # Initialize FC Block
        self.fc_block = FcBlock(num_outputs, self.fc_sizes)

    def forward(self, x):
        """Perform forward."""
        x = x
        x = self.convBlock(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x

    def VGG11(self, drop_vals=(0.3, 0.3, 0.4, 0.4, 0.5)):
        self.name = 'vgg11'
        self.convBlock = nn.Sequential()
        self.convBlock.add_module('conv1', VggBlock(3, 64, 1,  drop_vals[0]))
        self.convBlock.add_module('conv2', VggBlock(64, 128, 1,  drop_vals[1]))
        self.convBlock.add_module('conv3', VggBlock(128, 256, 2,  drop_vals[2]))
        self.convBlock.add_module('conv4', VggBlock(256, 512, 2,  drop_vals[3]))
        self.convBlock.add_module('conv5', VggBlock(512, 512, 2,  drop_vals[4]))

    def VGG13(self, drop_vals=(0.3, 0.3, 0.4, 0.4, 0.5)):
        self.name = 'vgg13'
        self.convBlock = nn.Sequential()
        self.convBlock.add_module('conv1', VggBlock(3, 64, 2,  drop_vals[0]))
        self.convBlock.add_module('conv2', VggBlock(64, 128, 2,  drop_vals[1]))
        self.convBlock.add_module('conv3', VggBlock(128, 256, 2,  drop_vals[2]))
        self.convBlock.add_module('conv4', VggBlock(256, 512, 2,  drop_vals[3]))
        self.convBlock.add_module('conv5', VggBlock(512, 512, 2,  drop_vals[4]))

    def VGG16(self, drop_vals=(0.3, 0.3, 0.4, 0.4, 0.5)):
        self.name = 'vgg16'
        self.convBlock = nn.Sequential()
        self.convBlock.add_module('conv1', VggBlock(3, 64, 2, drop_vals[0]))
        self.convBlock.add_module('mp1', self.mp_layer)
        self.convBlock.add_module('conv2', VggBlock(64, 128, 2, drop_vals[1]))
        self.convBlock.add_module('mp2', self.mp_layer)
        self.convBlock.add_module('conv3', VggBlock(128, 256, 3, drop_vals[2]))
        self.convBlock.add_module('mp3', self.mp_layer)
        self.convBlock.add_module('conv4', VggBlock(256, 512, 3, drop_vals[3]))
        self.convBlock.add_module('mp4', self.mp_layer)
        self.convBlock.add_module('conv5', VggBlock(512, 512, 3, drop_vals[4]))
        self.convBlock.add_module('mp5', self.mp_layer)

    def VGG19(self, drop_vals=(0.3, 0.3, 0.4, 0.4, 0.5)):
        self.name = 'vgg19'
        self.convBlock = nn.Sequential()
        self.convBlock.add_module('conv1', VggBlock(3, 64, 2,  drop_vals[0]))
        self.convBlock.add_module('conv2', VggBlock(64, 128, 2,  drop_vals[1]))
        self.convBlock.add_module('conv3', VggBlock(128, 256, 4,  drop_vals[2]))
        self.convBlock.add_module('conv4', VggBlock(256, 512, 4,  drop_vals[3]))
        self.convBlock.add_module('conv5', VggBlock(512, 512, 4,  drop_vals[4]))


class VGGmtl(nn.Module):

    def __init__(self):

        super(VGGmtl, self).__init__()
        self.name = 'vggmtl'
        self.loss_function = torch.nn.CrossEntropyLoss()

        self.B = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))

        self.conv1 = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),


            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.4),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),


            # Conv Layer block 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout2d(p=0.4),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout2d(p=0.4),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.conv2 = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),


            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.4),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),


            # Conv Layer block 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout2d(p=0.4),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout2d(p=0.4),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.fc_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, input):
        a, b = input[0], input[1]
        x1 = self.conv1(a)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.conv2(b)
        x2 = x2.view(x2.size(0), -1)
        x3 = x1.add(torch.mul(x2, self.B))
        x3 = self.fc_layer(x3)
        return x3
