import torch.nn as nn
import torch.nn.functional as F
import torch

class mnist_conv(nn.Module):
    def __init__(self, mean = 0.0, std = 1.0):
        super(mnist_conv, self).__init__()
        self.act_fn = nn.ReLU()
        #self.act_fn = nn.LeakyReLU(negative_slope=0.01)
        #self.act_fn = nn.Softplus()
        
        #
        self.conv = torch.nn.Conv2d
        self.linear = torch.nn.Linear
        self.mean = mean
        self.std = std

        self.layers1 = []
        self.layers2 = []
        self.layers1.append(self.conv(1, 64, 5))
        self.layers1.append(nn.MaxPool2d(2))
        self.layers1.append(self.act_fn)

        self.layers1.append(self.conv(64, 64, 5))
        self.layers1.append(nn.MaxPool2d(2))
        self.layers1.append(self.act_fn)

        self.layers1.append(nn.Flatten())

        self.layers2.append(self.linear(4 * 4 * 64, 128))
        self.layers2.append(self.act_fn)

        self.layers2.append(self.linear(128, 10))
        #self.layers2.append(torch.nn.Softmax(dim=1))        

        self.layers1 = nn.Sequential(*self.layers1)
        self.layers2 = nn.Sequential(*self.layers2)

    def forward(self, x):
        x = (x - self.mean)/self.std
        x = self.layers1(x)
        return self.layers2(x)
