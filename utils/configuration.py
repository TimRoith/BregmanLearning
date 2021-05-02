import torch
import torch.nn.functional as F
import math
import random
from itertools import cycle

# custom imports
from models.resnet import ResNet18
from models.mnist_conv import mnist_conv
from models.fully_connected import fully_connected
import models.aux_funs as maf
import regularizers as reg
from utils.datasets import data_set_info
import train
import optimizers as op

# numpy
import numpy as np


class Conf:
    def __init__(self, **kwargs):
        # CUDA settings
        self.use_cuda = kwargs.get('use_cuda', False)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.num_workers = kwargs.get('num_workers', 0)
        
        # dataset
        self.data_set = kwargs.get('data_set', "MNIST")
        self.data_file = kwargs.get('data_file', "data")
        self.train_split = kwargs.get('train_split', 1.0)
        # dataset info
        im_shape, mean, std = data_set_info(self.data_set,self.device)
        self.im_shape = im_shape
        self.data_set_mean = mean
        self.data_set_std = std
        # min and max of pictures
        self.x_min = 0.0
        self.x_max = 1.0
        

        
        # Loss function and norm
        self.loss = kwargs.get('loss', F.cross_entropy)
        
        # sparsity
        self.sparse_init = kwargs.get('sparse_init', 1.0)
        
        # -----------------------------
        self.goal_acc = kwargs.get('goal_accuracy', 0.0)
 
        # specification for Training
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 128)
        self.lr = kwargs.get('lr', 0.1)
            
           
        
        
class run:
    def __init__(self,**kwargs):
        self.num_runs = kwargs.get('num_runs', 5)
        self.run_iter = 0
        self.rep = 0
        
        # mu and sparse init
        self.mus = cycle(kwargs.get('mu', [0.1]))
        self.sparse_inits = cycle(kwargs.get('sparse_init', [0.01]))
        self.rs = cycle(kwargs.get('r', [[5,10]]))
        
        self.repitions = kwargs.get('repitions', 1)
        # history
        self.run_history = []
        
    def step(self):
        if self.rep == 0:
            self.run_iter += 1
        
        if self.run_iter <= self.num_runs:
            if self.rep == 0 :
                self.run_history.append({})
                self.mu = next(self.mus)
                self.sparse_init = next(self.sparse_inits)
                self.r = next(self.rs)
               
            # update repition counter
            self.rep = (self.rep + 1)%self.repitions

            return True
        else:
            return False
        
    def add_history(self, hist, name):
        for key in hist:
            loc_key = name + "_"+ key
            rs_loc = self.run_history[self.run_iter-1]
            if loc_key not in rs_loc:
                rs_loc[loc_key] = torch.zeros((len(hist[key]),self.repitions))
            
            # add to history   
            rs_loc[loc_key][:,self.rep-1] = torch.FloatTensor(hist[key])
             
            
        

# -----------------------------------------------------------------------------------
# no regularization
# -----------------------------------------------------------------------------------
def plain_example(data_file, use_cuda=False, num_workers=None):
    if use_cuda and num_workers is None:
        num_workers = 4
    else:
        num_workers = 0
    
    conf_args = {'lamda':0.0,'data_file':data_file, 'use_cuda':use_cuda, 'train_split':0.9, 'num_workers':num_workers,
                 'regularization': "none", 'activation_function':"sigmoid"}

    # get configuration
    conf = Conf(**conf_args)
    
    return conf

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# -----------------------------------------------------------------------------------
# Fashion_MNIST_CONV
# -----------------------------------------------------------------------------------
def mnist_example(data_file, use_cuda=False, num_workers=None, 
                  mu=0.0, sparse_init=1.0, r = [5,10], lr=0.1, optim = "PSGD", beta = 0.0, delta = 1.0):
    if use_cuda and num_workers is None:
        num_workers = 4
    else:
        num_workers = 0
    
    conf_args = {'data_set': "MNIST", 'data_file':data_file, 
                 'use_cuda':use_cuda, 'train_split':0.95, 'num_workers':num_workers, 'epochs':100, 
                 'sparse_init':sparse_init}

    # get configuration
    conf = Conf(**conf_args)

    
    # -----------------------------------------------------------------------------------
    # define the model and an instance of the best model class
    # -----------------------------------------------------------------------------------
    sizes = [784, 200, 80, 10]
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
    maf.sparsify_(model, conf.sparse_init)
    model = model.to(conf.device)
    
    # -----------------------------------------------------------------------------------
    # Get access to different model parameters
    # -----------------------------------------------------------------------------------
    weights_linear = maf.get_weights_linear(model)
    biases = maf.get_bias(model)
    
    # -----------------------------------------------------------------------------------
    # Initialize optimizer
    # -----------------------------------------------------------------------------------
    if optim == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=beta)
    elif optim == "LinBreg":
        opt = op.LinBreg([{'params': weights_linear, 'lr' : lr, 'reg' : reg.reg_l1(mu=mu), 'momentum':beta, 'delta':delta},
                          {'params': biases, 'lr': lr, 'momentum':beta}])
    elif optim == "adam":
        opt = op.AdamBreg([{'params': weights_linear, 'lr' : lr, 'reg' : reg.reg_l1(mu=mu)},
                          {'params': biases, 'lr': lr}])
    elif optim == "PSGD":
        opt = op.ProxSGD([{'params': weights_linear, 'lr' : lr, 'reg' : reg.reg_l1(mu=mu)},
                          {'params': biases, 'lr': lr}])
        
        
    
    return conf, model, best_model, opt
# -----------------------------------------------------------------------------------
# Fashion_MNIST_CONV
# -----------------------------------------------------------------------------------
def fashion_mnist_example(data_file, use_cuda=False, num_workers=None, mu=[0.0, 0.0], sparse_init=1.0,
                          r = [5,10], lr=0.1, optim = "SGD", beta = 0.0, delta = 1.0):
    if use_cuda and num_workers is None:
        num_workers = 4
    else:
        num_workers = 0
    
    conf_args = {'data_set': "Fashion-MNIST", 'data_file':data_file, 
                 'use_cuda':use_cuda, 'train_split':0.95, 'num_workers':num_workers,
                 'sparse_init':sparse_init}

    # get configuration
    conf = Conf(**conf_args)

    
    # -----------------------------------------------------------------------------------
    # define the model and an instance of the best model class
    # -----------------------------------------------------------------------------------
    model = mnist_conv(mean = conf.data_set_mean, std = conf.data_set_std)
    best_model = train.best_model(mnist_conv(mean = conf.data_set_mean, std = conf.data_set_std).to(conf.device), goal_acc = conf.goal_acc)
    
    
    maf.sparse_bias_uniform_(model, 0,r[0])
    maf.sparse_bias_uniform_(model, 0,r[0],ltype=torch.nn.Conv2d)
    maf.sparse_weight_normal_(model, r[1])
    #
    maf.sparsify_(model, conf.sparse_init)
    model = model.to(conf.device)
    
    # -----------------------------------------------------------------------------------
    # Get access to different model parameters
    # -----------------------------------------------------------------------------------
    weights_conv = maf.get_weights_conv(model)
    weights_linear = maf.get_weights_linear(model)
    biases = maf.get_bias(model)
    
    # -----------------------------------------------------------------------------------
    # Initialize optimizer
    # -----------------------------------------------------------------------------------
    if optim == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=beta)
    elif optim == "LinBreg":
        opt = op.LinBreg([{'params': weights_conv, 'lr' : lr, 'reg' : reg.reg_l1_l2_conv(mu=mu[0]), 'momentum':beta},
                          {'params': weights_linear, 'lr' : lr, 'reg' : reg.reg_l1(mu=mu[1]), 'momentum':beta},
                          {'params': biases, 'lr': lr, 'momentum':beta}])
    elif optim == "ProxSGD":
        opt = op.ProxSGD([{'params': weights_conv, 'lr' : lr, 'reg' : reg.reg_l1_l2_conv(mu=mu[0]), 'momentum':beta},
                          {'params': weights_linear, 'lr' : lr, 'reg' : reg.reg_l1(mu=mu[1]), 'momentum':beta},
                          {'params': biases, 'lr': lr, 'momentum':beta}])            
    elif optim == "AdaBreg":
        opt = op.AdamBreg([{'params': weights_conv, 'lr' : lr, 'reg' : reg.reg_l1_l2_conv(mu=mu[0])},
                           {'params': weights_linear, 'lr' : lr, 'reg' : reg.reg_l1(mu=mu[1])},
                           {'params': biases, 'lr': lr}])
    else:
        raise ValueError("Unknown Optimizer specified")
    
    return conf, model, best_model, opt

# -----------------------------------------------------------------------------------
# Fashion_MNIST_CONV
# -----------------------------------------------------------------------------------
def cifar10_example(data_file, use_cuda=False, num_workers=None, 
                    r = [1.0,1.0], mu=[0.0, 0.0], sparse_init=1.0, lr= 0.1, beta=0.0, optim="SGD",delta=1.0):
    if use_cuda and num_workers is None:
        num_workers = 4
    else:
        num_workers = 0
    
    conf_args = {'data_set': "CIFAR10", 'data_file':data_file, 
                 'use_cuda':use_cuda, 'train_split':0.95, 'num_workers':num_workers,
                 'sparse_init':sparse_init}

    # get configuration
    conf = Conf(**conf_args)

    
    # -----------------------------------------------------------------------------------
    # define the model and an instance of the best model class
    # -----------------------------------------------------------------------------------
    model = ResNet18(mean = conf.data_set_mean, std = conf.data_set_std)
    best_model = train.best_model(ResNet18(mean = conf.data_set_mean, std = conf.data_set_std).to(conf.device), goal_acc = conf.goal_acc)
    
    # sparsify
    maf.sparse_bias_uniform_(model, 0,r[0])
    maf.sparse_weight_normal_(model, r[1])
    #
    maf.sparsify_(model, conf.sparse_init)
    model = model.to(conf.device)
    
    # -----------------------------------------------------------------------------------
    # Get access to different model parameters
    # -----------------------------------------------------------------------------------
    weights_conv = maf.get_weights_conv(model)
    weights_linear = maf.get_weights_linear(model)
    weights_batch = maf.get_weights_batch(model)
    biases = maf.get_bias(model)
    
    # -----------------------------------------------------------------------------------
    # Initialize optimizer
    # -----------------------------------------------------------------------------------
    if optim == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=beta)
    elif optim == "LinBreg":
        opt = op.LinBreg([{'params': weights_conv, 'lr' : lr, 'reg' : reg.reg_l1_l2_conv(mu=mu[0]), 'momentum':beta},
                          {'params': weights_linear, 'lr' : lr, 'reg' : reg.reg_l1(mu=mu[1]), 'momentum':beta},
                          {'params': weights_batch, 'lr' : lr, 'momentum':beta},
                          {'params': biases, 'lr': lr, 'momentum':beta}])
    elif optim == "AdaBreg":
        opt = op.AdamBreg([{'params': weights_conv, 'lr' : lr, 'reg' : reg.reg_l1_l2_conv(mu=mu[0])},
                           {'params': weights_linear, 'lr' : lr, 'reg' : reg.reg_l1(mu=mu[1])},
                           {'params': weights_batch, 'lr' : lr, 'momentum':beta},
                           {'params': biases, 'lr': lr}])
    else:
        raise ValueError("Unknown Optimizer specified")
    
    return conf, model, best_model, opt
            
