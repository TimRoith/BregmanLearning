import torch.nn as nn
import torch.nn.functional as F
import torch
from itertools import cycle

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class fully_connected(nn.Module):
    def __init__(self, sizes, act_fn):
        super(fully_connected, self).__init__()
        
        self.act_fn = act_fn
        layer_list = [Flatten()]
        for i in range(len(sizes)-1):
            layer_list.append(nn.Linear(sizes[i], sizes[i+1]))
            layer_list.append(self.act_fn)
            
        self.layers = nn.Sequential(*layer_list)
        
        
    def forward(self, x):
        return self.layers(x)
    
    
def init_weight_bias_normal(m):
    if type(m) == nn.Linear:
        m.weight.data = torch.randn_like(m.weight.data)
        m.bias.data = torch.randn_like(m.bias.data)
        
        
def init_sparse(sparsity):
    if isinstance(sparsity, list):
        s_iter = cycle(sparsity)
    else:
        s_iter = cycle([sparsity])
      
    def init_sparse_from_it(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            s_loc = next(s_iter)
            # number of zeros
            n = int(s_loc*m.weight.numel())
            # initzialize mask
            mask = torch.zeros_like(m.weight)
            row_idx = torch.randint(low=0,high=mask.shape[0],size=(n,))
            col_idx = torch.randint(low=0,high=mask.shape[1],size=(n,))
            # fill with ones at random indices
            mask[row_idx, col_idx] = 1.
            m.weight.data.mul_(mask)
            
    return init_sparse_from_it

def he_sparse_(tensor, sparsity):
    rows, cols = tensor.shape
    num_zeros = int(sparsity * rows)

    with torch.no_grad():
        tensor.normal_(0, cols*(1-sparsity))
        for col_idx in range(cols):
            row_indices = torch.randperm(rows)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor


def print_sparsity(M,print_all=True):
    s =""
    s_list =[]
    n=""
    n_list=[]
    sp=0
    numel=0
    for m in M:
        if isinstance(m, torch.nn.Linear):
            a = m.weight
            numel_loc = a.data.numel()
            numel += numel_loc
            
            sp_loc = torch.count_nonzero(a.data).item()
            sp += sp_loc
            s += str(sp_loc/numel_loc) + " "
            s_list.append(sp_loc/numel_loc)
            n += str(torch.count_nonzero(torch.sum(torch.abs(a.data),axis=1)).item()) + "/" + str(a.data.shape[0]) + " "
            n_list.append(torch.count_nonzero(torch.sum(torch.abs(a.data),axis=1)).item()/a.data.shape[0])
        elif isinstance(m, torch.nn.Conv2d):
            a = m.weight
            numel_loc = a.data.numel()
            numel += numel_loc
            
            sp_loc = torch.count_nonzero(a.data).item()
            sp += sp_loc
            s += str(sp_loc/numel_loc) + " "
    
            
    print(50*'-')
    if print_all:
        print('Weight Sparsity:', s)
        print('Active Nodes:', n)
    print('Total percentage of used weights:',(sp/numel))
    
    return s_list, n_list, sp/numel


def conv_sparsity(model):
    nnz = 0
    total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            s = m.weight.shape
            w = m.weight.view(s[0]*s[1], s[2]*s[3])
            nnz += torch.count_nonzero(torch.sum(w,dim=1)>0).item()
            total += s[0] * s[1]
    #
    return nnz/total
        

def get_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            yield m.weight
        else:
            continue
            
def get_weights_conv(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            yield m.weight
        else:
            continue
            
def get_weights_linear(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            yield m.weight
        else:
            continue
            
def get_bias(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            yield m.bias
        else:
            continue
            
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
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
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
class mnist_conv(nn.Module):
    def __init__(self):
        super(mnist_conv, self).__init__()
        self.act_fn = nn.ReLU
        self.conv = torch.nn.Conv2d
        self.linear = torch.nn.Linear

        self.layers1 = []
        self.layers2 = []
        self.layers1.append(self.conv(1, 64, 5))
        self.layers1.append(nn.MaxPool2d(2))
        self.layers1.append(self.act_fn())

        self.layers1.append(self.conv(64, 64, 5))
        self.layers1.append(nn.MaxPool2d(2))
        self.layers1.append(self.act_fn())

        self.layers1.append(nn.Flatten())

        self.layers2.append(self.linear(4 * 4 * 64, 128))
        self.layers2.append(self.act_fn())

        self.layers2.append(self.linear(128, 10))

        self.layers1 = nn.Sequential(*self.layers1)
        self.layers2 = nn.Sequential(*self.layers2)

    def forward(self, x):
        x = (x - self.mean)/self.std
        x = self.layers1(x)
        return self.layers2(y)
        
