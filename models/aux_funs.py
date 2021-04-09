import torch.nn as nn
import torch.nn.functional as F
import torch
from itertools import cycle
  
    
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
            nnz += torch.count_nonzero(torch.norm(w,p=1,dim=1)>0).item()
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
            
            
