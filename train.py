import torch
import torch.nn.utils.prune as prune
import models.aux_funs as maf

# train step
def train_step(conf, model, opt, train_loader, verbosity = 1):
    model.train()
    acc = 0
    tot_loss = 0.0
    tot_steps = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        # get batch data
        x, y = x.to(conf.device), y.to(conf.device)
        opt.zero_grad()
        logits = model(x)
        loss = conf.loss(logits, y)
        
        if hasattr(conf,"weight_reg"):
            loss += conf.weight_reg(model)
            
        
        loss.backward()
        opt.step()
        
        # for classification tasks we want to evaluate the accuracy
        if conf.eval_acc:
            acc += (logits.max(1)[1] == y).sum().item()
        
        tot_loss += loss.item()
        tot_steps += y.shape[0]

    # print the current accuracy and loss
    if verbosity > 0: 
        print(50*"-")
        print('Train Accuracy:', acc/tot_steps)
        print('Train Loss:', tot_loss)
    return {'loss':tot_loss, 'acc':acc/tot_steps}




# validation step
def validation_step(conf, model, opt, validation_loader, verbosity = 1):
    acc = 0.0
    loss = 0.0
    tot_steps = 0
    # -------------------------------------------------------------------------
    # evaluate on validation set
    if not validation_loader is None:
        for batch_idx, (x, y) in enumerate(validation_loader):
            # get batch data
            x, y = x.to(conf.device), y.to(conf.device)

             # evaluate model on batch
            logits = model(x)

            # Get classification loss
            c_loss = conf.loss(logits, y)
            
            if conf.eval_acc:
                acc += (logits.max(1)[1] == y).sum().item()
            loss += c_loss.item()
            tot_steps += y.shape[0]
        tot_acc = acc/tot_steps
    else:
        tot_acc = 0.0
            
    # ------------------------------------------------------------------------
    # evaluate sparsity
    conv_sparse = maf.conv_sparsity(model)
    linear_sparse = maf.linear_sparsity(model)
    net_sparse = maf.net_sparsity(model)
    node_sparse = maf.node_sparsity(model)
    
    # ------------------------------------------------------------------------
    # evaluate regularizers of opt and append to history
    reg_eval = getattr(opt, "evaluate_reg", None)
    if callable(reg_eval):
        reg_vals = opt.evaluate_reg()
    else:
        reg_vals = []
         
    # print values
    if verbosity > 0: 
        print(50*"-")
        print('Validation Accuracy:', tot_acc)
        print('Non-zero kernels:', conv_sparse)
        print('Linear sparsity:', linear_sparse)
        print('Overall sparsity:', net_sparse)
        print('Node sparsity:', node_sparse)
        
        
        print('Regularization values per group:', reg_vals)
    return {'loss':loss, 'acc':tot_acc, 'conv_sparse':conv_sparse, 'linear_sparse':linear_sparse,
            'reg_vals':reg_vals,'node_sparse':node_sparse}




# test step
def test(conf, model, test_loader, verbosity=1):
    model.eval()
    acc = 0
    tot_steps = 0
    loss = 0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            # get batch data
            x, y = x.to(conf.device), y.to(conf.device)
            # evaluate
            pred = model(x)
            if conf.eval_acc:
                acc += (pred.max(1)[1] == y).sum().item()
            
            c_loss = conf.loss(pred, y)
            loss += c_loss.item()
            
            tot_steps += y.shape[0]
    
    # print accuracy
    if verbosity > 0: 
        print(50*"-")
        print('Test Accuracy:', acc/tot_steps)
    return {'acc':acc/tot_steps, 'loss':loss}

# ------------------------------------------------------------------------
# Pruning step
def prune_step(model, a1 = 0.01, a2 = 0.01, conv_group=True):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and conv_group:
            prune.ln_structured(module, name='weight', amount=a1,n=2,dim=0)
            prune.ln_structured(module, name='weight', amount=a1,n=2,dim=1)
            
            #prune.remove(module, name='weight')
        elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=a2)
            
            #prune.remove(module, name='weight')
            
        
            


class best_model:
    '''saves the best model'''
    def __init__(self, best_model=None, gamma = 0.0, goal_acc = 0.0):
        # stores best seen score and model
        self.best_score = 0.0
        
        # if specified, a copy of the model gets saved into this variable
        self.best_model = best_model

        # score function
        def score_fun(train_acc, test_acc):
            return gamma * train_acc + (1-gamma) * test_acc + (train_acc > goal_acc)
        self.score_fun = score_fun
        
    
    def __call__(self, train_acc, val_acc, model=None):
        # evaluate score
        score = self.score_fun(train_acc, val_acc)
        if score >= self.best_score:
            self.best_score = score
            # store model
            if self.best_model is not None:
                self.best_model.load_state_dict(model.state_dict())