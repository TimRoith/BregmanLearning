# Various torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# custom packages
import models.aux_funs as maf
import optimizers as op
import regularizers as reg
import train
from utils.configuration import Conf
from utils.datasets import get_data_set

# specify path to data
data_file = "../Data"

# -----------------------------------------------------------------------------------
# Set up configuration
# -----------------------------------------------------------------------------------
conf_args = {'data_file':data_file, 'data_set': "Fashion-MNIST", 'use_cuda':True, 'train_split':1.0, 'num_workers':4}
conf = Conf(**conf_args)

# get train, validation and test loader
train_loader, valid_loader, test_loader = get_data_set(conf)

# -----------------------------------------------------------------------------------
# define the model and an instance of the best model class
# -----------------------------------------------------------------------------------
from models.mnist_conv import mnist_conv
model = mnist_conv(mean = conf.data_set_mean, std = conf.data_set_std)
best_model = train.best_model(mnist_conv(mean = conf.data_set_mean, std = conf.data_set_std).to(conf.device), goal_acc = conf.goal_acc)

# sparsify
maf.sparsify_(model, 0.01)
maf.sparse_he_(model, 0.1)
model = model.to(conf.device)
# -----------------------------------------------------------------------------------
# Get acces to different model parameters
# -----------------------------------------------------------------------------------
weights_conv = maf.get_weights_conv(model)
weights_linear = maf.get_weights_linear(model)
biases = maf.get_bias(model)

# -----------------------------------------------------------------------------------
# Initialize optimizer and learning rate scheduler
# -----------------------------------------------------------------------------------
optim = "vanilla-breg"
if optim == "vanilla":
    opt = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9)
elif optim == "vanilla-breg":
    beta = 0.9
    lr = 0.01/(1-beta)
    mu = 0.1
    opt = op.LinBreg([{'params': weights_conv, 'lr' : lr, 'reg' : reg.reg_l1_l2_conv(mu=0.1), 'momentum':beta},
                      {'params': weights_linear, 'lr' : lr, 'reg' : reg.reg_l1(mu=mu), 'momentum':beta},
                      {'params': biases, 'lr': lr, 'momentum':beta}])
elif optim == "adam":
    opt = op.AdamBreg([{'params': weights, 'lr' : 1e-3, 'reg' : reg.reg_l1(mu=1.0)},
                    {'params': biases, 'lr': 1e-3}])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
# initalize history
# -----------------------------------------------------------------------------------
tracked = ['loss', 'acc', 'l1-norm', 'conv_sparsity']
train_history = {key: [] for key in tracked}
val_history = {key: [] for key in tracked}


# -----------------------------------------------------------------------------------
# train the model
# -----------------------------------------------------------------------------------
for epoch in range(conf.epochs):
    print(25*"<>")
    print(50*"|")
    print(25*"<>")
    print('Epoch:', epoch)
    
    # ------------------------------------------------------------------------
    # train step, log the accuracy and loss
    train_data = train.train_step(conf, model, opt, train_loader)
    
    # update history
    for key in tracked:
        if key in train_data:
            train_history[key].append(train_data[key])
    
    # ------------------------------------------------------------------------
    # validation step
    if not (valid_loader is None):
        val_data = train.validation_step(conf, model, valid_loader)
    
    # ------------------------------------------------------------------------
    # evaluate sparsity
    conv_sparse = maf.conv_sparsity(model)
    print('Non-zero kernels:', conv_sparse)
    train_history['conv_sparsity'] = conv_sparse
    
    # get the current sparsity and node usage
    # s_list, n_list, s_tot = cm.print_sparsity(model.modules(),print_all=False)
    # s_tots.append(s_tot)
    #     for i in range(0,len(s)):
    #         s[i].append(s_list[i])
    #         n[i].append(n_list[i])
    
    # ------------------------------------------------------------------------
    # Evaluate L1 norm and append to history
    reg_vals = opt.evaluate_reg()
    print('Regularization values per group:', reg_vals)
    for i,reg_val in enumerate(reg_vals):
        key = "group_reg_" + str(i)
        if key in train_history:
            train_history[key].append(reg_val)
        else:
            train_history[key] = [reg_val]
    
    scheduler.step(train_data['loss'])
    print("Learning rate:",opt.param_groups[0]['lr'])
