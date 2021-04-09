import torch
import matplotlib.pyplot as plt
import math

def plot_conv_mat(m, idx=0):
    if isinstance(m, torch.nn.Conv2d):
        s = m.weight.shape
        num_mat = s[0]
        ax_dim = math.ceil(math.sqrt(num_mat))
        
        fig, axs = plt.subplots(ax_dim, ax_dim)
        plt.axis('off')
        
        for i in range(num_mat):
            row = i%ax_dim
            col = i//ax_dim
            
            axs[row][col].matshow(m.weight[i,idx,::].cpu().detach())
            axs[row][col].set_axis_off()
