import torch
import torch.nn.functional as F
from torchvision import transforms
import math
import random
from itertools import cycle

# custom imports
from models.resnet import ResNet18
from models.mnist_conv import mnist_conv
from models.fully_connected import fully_connected
import models.aux_funs as maf
import regularizers as reg
from utils.datasets import data_set_info, add_noise
import train
import optimizers as op

# -----
import os.path
import csv

# numpy
import numpy as np


class Conf:
    def __init__(self, **kwargs):
        # ----------------------------------------
        # set defaults
        # ----------------------------------------
        # cuda
        self.use_cuda = False
        self.num_workers = 0
        self.cuda_device = 0
        
        # dataset
        self.data_set = "MNIST"
        self.data_file = ""
        self.train_split = 1.0
        self.download = False
        
        # loss function
        self.loss = F.cross_entropy
        
        # misc
        self.eval_acc = True
 
        # specification for Training
        self.epochs = 100
        self.batch_size = 128
        self.lr = 0.1
        self.sparse_init = 1.0
        
        # ----------------------------------------
        # set all kwargs
        # ----------------------------------------
        for key, value in kwargs.items():
            setattr(self, key, value)
        # ----------------------------------------
        
        # Set device
        self.device = torch.device("cuda"+":"+str(self.cuda_device) if self.use_cuda else "cpu")
        
        # additonal dataset info
        im_shape, mean, std = data_set_info(self.data_set,self.device)
        self.im_shape = im_shape
        self.data_set_mean = mean
        self.data_set_std = std
        self.x_min = 0.0
        self.x_max = 1.0

    def write_to_csv(self):
        idx = 0
        log_file = 'ex_results/'+self.super_type+'/'+self.name+'_'+str(idx)+'.csv'
        while os.path.isfile(log_file):
            idx += 1
            log_file = 'ex_results/'+self.super_type+'/'+self.name+'_'+str(idx)+'.csv'
    
        self.ID = idx
    
        with open(log_file, mode='w') as res_file:
            res_writer = csv.writer(res_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            v_conf = vars(self)

            key_loc = list(v_conf.keys())
            res_writer.writerow(key_loc)

            values = []
            for key in key_loc:
                values.append(str(v_conf[key]))
            res_writer.writerow(values)
        
class run:
    def __init__(self,params):
        self.iter = 0
        self.params = []
        for i in range(len(params)):
            p = params[i]
            reps = p.get('reps',1)

            for j in range(reps):
                # new dictionary to use for local parameters
                p_loc = p.copy()
                p_loc['random_seed'] = p.get('random_seed',0) + j
                
                self.params.append(p_loc)
                
        self.num_runs = len(self.params)
        # ----------------------------------------        
        # history
        self.history = []
        
    def step(self,conf=None):
        if self.iter < self.num_runs:
            if not (conf is None):
                for key in self.params[self.iter]:
                    setattr(conf, key,self.params[self.iter][key])
            # ----------------------------------------
            self.history.append({})
            self.iter += 1
            return True
        else:
            return False
        
    def add_history(self, hist, name):
        for key in hist:
            loc_key = name + "_"+ key
            
            # add to history   
            self.history[self.iter-1][loc_key] = hist[key]
             
            
# -----------------------------------------------------------------------------------
# fix a specific seed
# -----------------------------------------------------------------------------------
def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# -----------------------------------------------------------------------------------
# MNIST_AUTOENCODER
# -----------------------------------------------------------------------------------
def mnist_autoencoder_example(data_file, use_cuda=False, num_workers=None, 
                  mu=0.0, sparse_init=1.0, r = [5,10], lr=0.1, optim = "SGD", beta = 0.0, delta = 1.0):
    if use_cuda and num_workers is None:
        num_workers = 4
    else:
        num_workers = 0
        
    
    conf_args = {'data_set': "Encoder-MNIST", 'data_file':data_file, 
                 'use_cuda':use_cuda, 'train_split':0.95, 'num_workers':num_workers, 'epochs':100, 
                 'sparse_init':sparse_init, 'eval_acc':False}
    
    # get configuration
    conf = Conf(**conf_args)
    
#     def reshaped_mse_loss(x,y):
#         kernel_size = 5
#         sigma=(0.1, 2.0)
#         x_aug = add_noise(std=0.2,device=conf.device)(x)
#         return torch.nn.MSELoss()(x_aug,x.view(-1,28*28))

    def reshaped_mse_loss(x,y):
            return torch.nn.MSELoss()(x,y.view(-1,28*28))

    conf.loss = reshaped_mse_loss

    
    # -----------------------------------------------------------------------------------
    # define the model and an instance of the best model class
    # -----------------------------------------------------------------------------------
    sizes = 7*[784]
    #act_fun = torch.nn.Sigmoid()
    act_fun = torch.nn.ReLU()
    #act_fun = torch.nn.Softplus(beta=1, threshold=20)
    #act_fun = torch.nn.LeakyReLU(0.2)
    
    model = fully_connected(sizes, act_fun, mean = conf.data_set_mean, std = conf.data_set_std)
    best_model = train.best_model(fully_connected(sizes, act_fun, mean = conf.data_set_mean, 
                                                  std =conf.data_set_std).to(conf.device), goal_acc = conf.goal_acc)
    
    # sparsify
    #maf.sparse_he_(model, 5.0)
    maf.sparse_bias_uniform_(model, 0,r[0])
    #maf.bias_constant_(model,r[0])
    
    maf.sparse_weight_normal_(model, r[1])
    maf.sparsify_(model, conf.sparse_init, row_group = True)
    model = model.to(conf.device)
    
    # -----------------------------------------------------------------------------------
    # Get access to different model parameters
    # -----------------------------------------------------------------------------------
    weights_linear = maf.get_weights_linear(model)
    biases = maf.get_bias(model)
    
    # -----------------------------------------------------------------------------------
    # Initialize optimizer
    # -----------------------------------------------------------------------------------
    reg1 = reg.reg_l1_l2(mu=mu)
    #reg1 =  reg.reg_l1(mu=mu)
    
    if optim == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=beta)
    elif optim == "LinBreg":
        opt = op.LinBreg([{'params': weights_linear, 'lr' : lr, 'reg' : reg1, 'momentum':beta, 'delta':delta},
                          {'params': biases, 'lr': lr, 'momentum':beta}])
    elif optim == "AdaBreg":
        opt = op.AdamBreg([{'params': weights_linear, 'lr' : lr, 'reg' : reg1},
                           {'params': biases, 'lr': lr}])
    elif optim == "ProxSGD":
        opt = op.ProxSGD([{'params': weights_linear, 'lr' : lr, 'reg' : reg1, 'momentum':beta,'delta':delta},
                          {'params': biases, 'lr': lr, 'momentum':beta}])  
    elif optim == "L1SGD":
        def weight_reg(model):         
            loss1 = 0
            for w in maf.get_weights_linear(model):
                loss1 += reg1(w)
                
            return loss1
        
        conf.weight_reg = weight_reg
        
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=beta)
    
    return conf, model, best_model, opt
            
