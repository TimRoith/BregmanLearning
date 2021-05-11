import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

# custom packages
import models.aux_funs as maf
import optimizers as op
import regularizers as reg
import train
import math
import utils.configuration as cf
from utils.datasets import get_data_set

# specify path to data
data_file = "../Data"

# -----------------------------------------------------------------------------------
# Parameters for different runs
# -----------------------------------------------------------------------------------
sis = [0.01]
mus = [0.1]
runs = cf.run(**{'mu':mus, 'sparse_init':sis, 'num_runs':max(len(sis),len(mus)), 'repitions':1})
# -----------------------------------------------------------------------------------
# Loop training over multiple runs
# -----------------------------------------------------------------------------------
cf.seed_torch(0)
while runs.step():
    mu = runs.mu
    sparse_init = runs.sparse_init
    r = [1,1,1]
    ll=0.5
    rr = ll/math.sqrt(sparse_init+0.001)
    r[0] = 1
    r[1] = rr
    r[2] = rr
    
    print("Using mu:",mu)
    print("Using r:",r[0],r[1])
    print("Repition:", runs.rep)
    # -----------------------------------------------------------------------------------
    # Set up configuration, obtain model and optimizer
    # -----------------------------------------------------------------------------------
    conf, model, best_model, opt = cf.cifar10_example(data_file, use_cuda=True, cuda_device=1,
                                                    mu=mu,sparse_init=sparse_init, r = r,lr=lr,optim="AdaBreg",
                                                    beta=0.9,delta=50.0,conv_group=False)
    
    # get train, validation and test loader
    train_loader, valid_loader, test_loader = get_data_set(conf)

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5,threshold=0.01)
    lsc = op.lamda_scheduler(opt,0, increment = 0.005, cooldown=5, target_sparse=0.1)

    # -----------------------------------------------------------------------------------
    # initalize history
    # -----------------------------------------------------------------------------------
    tracked = ['loss', 'acc', 'l1-norm', 'conv_sparse', 'linear_sparse']
    train_history = {key: [] for key in tracked}
    val_history = {key: [] for key in tracked}
    print(maf.linear_sparsity(model))
    print(maf.conv_sparsity(model))
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
        val_data = train.validation_step(conf, model, opt, valid_loader, opt_reg_eval=False)

        # update history
        for key in tracked:
            if key in val_data:
                val_history[key].append(val_data[key])

        for i,reg_val in enumerate(val_data['reg_vals']):
            key = "group_reg_" + str(i)
            if key in val_history:
                val_history[key].append(reg_val)
            else:
                val_history[key] = [reg_val]

        #scheduler step
        scheduler.step(train_data['loss'])
        #lsc(val_data['conv_sparse'])
        
        
        print("Learning rate:",opt.param_groups[0]['lr'])
        best_model(train_data['acc'], val_data['acc'], model=model)

    test_data = train.test(conf, best_model.best_model, test_loader)
    val_history['test_acc'] = [test_data['acc']]
    runs.add_history(train_history, "train")
    runs.add_history(val_history, "val")
