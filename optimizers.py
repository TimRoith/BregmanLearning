import torch
import math
import regularizers as reg

class LinBreg(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,reg=reg.reg_none(), delta=1.0, momentum=0.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
            
        defaults = dict(lr=lr, reg=reg, delta=delta, momentum=momentum)
        super(LinBreg, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            delta = group['delta']
            # define regularizer for this group
            reg = group['reg'] 
            step_size = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                # get grad and state
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # get prox
                    # initialize subgradients
                    state['sub_grad'] = self.initialize_sub_grad(p,reg, delta)
                    state['momentum_buffer'] = None
                # -------------------------------------------------------------
                # update scheme
                # -------------------------------------------------------------
                # get the current sub gradient
                sub_grad = state['sub_grad']
                # update on the subgradient
                if momentum > 0.0: # with momentum
                    mom_buff = state['momentum_buffer']
                    if state['momentum_buffer'] is None:
                        mom_buff = torch.zeros_like(grad)
 
                    mom_buff.mul_(momentum)
                    mom_buff.add_((1-momentum)*step_size*grad) 
                    state['momentum_buffer'] = mom_buff
                    #update subgrad
                    sub_grad.add_(-mom_buff)
                                                            
                else: # no momentum
                    sub_grad.add_(-step_size * grad)
                # update step for parameters
                p.data = reg.prox(delta * sub_grad, delta)
        
    def initialize_sub_grad(self,p, reg, delta):
        p_init = p.data.clone()
        return 1/delta * p_init + reg.sub_grad(p_init)
    
# ------------------------------------------------------------------------------------------------------    
class ProxSGD(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,reg=reg.reg_none()):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
            
        defaults = dict(lr=lr, reg=reg)
        super(ProxSGD, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            # define regularizer for this group
            reg = group['reg'] 
            step_size = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                # get grad and state
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    
                # -------------------------------------------------------------
                # update scheme
                # -------------------------------------------------------------               
                # gradient steps
                p.data.add_(-step_size * grad)
                # proximal step
                p.data = reg.prox(p.data, step_size)
                   
# ------------------------------------------------------------------------------------------------------           
class AdamBreg(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,reg=reg.reg_none(), delta=1.0, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
            
        defaults = dict(lr=lr, reg=reg, delta=delta, betas=betas, eps=eps)
        super(AdamBreg, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            delta = group['delta']
            # get regularizer for this group
            reg = group['reg']
            # get parameters for adam
            step_size = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                # get grad and state
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # get prox
                    # initialize subgradients
                    state['sub_grad'] = self.initialize_sub_grad(p,reg, delta)
                    state['exp_avg'] = torch.zeros_like(state['sub_grad'])
                    state['exp_avg_sq'] = torch.zeros_like(state['sub_grad'])
                # -------------------------------------------------------------
                # update scheme
                # -------------------------------------------------------------
                # update step
                state['step'] += 1
                step = state['step']
                # get the current sub gradient and averages
                sub_grad = state['sub_grad']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                # define bias correction factors
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # denominator in the fraction
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                # step size in adam update
                alpha = step_size / bias_correction1
                
                # update subgrad
                sub_grad.addcdiv_(exp_avg, denom, value=-step_size)
                
                # update step for parameters
                p.data = reg.prox(delta * sub_grad, delta)
        
    def initialize_sub_grad(self,p, reg, delta):
        p_init = p.data.clone()
        return 1/delta * p_init + reg.sub_grad(p_init)
