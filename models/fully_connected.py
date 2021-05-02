import torch.nn as nn
import torch.nn.functional as F
import torch

# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)

class fully_connected(nn.Module):
    def __init__(self, sizes, act_fn, mean=0.0, std=1.0):
        super(fully_connected, self).__init__()
        self.mean = mean
        self.std = std
        
        self.act_fn = act_fn
        layer_list = [nn.Flatten()]
        for i in range(len(sizes)-1):
            layer_list.append(nn.Linear(sizes[i], sizes[i+1]))
            layer_list.append(self.act_fn)
            
        self.layers = nn.Sequential(*layer_list)
        
        
    def forward(self, x):
        x = (x-self.mean)/self.std
        return self.layers(x)
