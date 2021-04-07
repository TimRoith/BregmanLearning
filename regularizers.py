import torch

class reg_none:
    def __call__(self, x):
        return 0
    
    def prox(self, x, delta=1.0):
        return x
    
    def sub_grad(self, v):
        return torch.zeros_like(v)
    
class reg_l1:
    def __init__(self, mu=1.0):
        self.mu = mu
        
    def __call__(self, x):
        return torch.norm(x, p=1)
        
    def prox(self, x, delta=1.0):
        return torch.sign(x) * torch.clamp(torch.abs(x) - (delta * self.mu),min=0)
        
    def sub_grad(self, v):
        return self.mu * torch.sign(v)
    
    
    
class reg_l1_l2:
    def __init__(self, mu=1.0):
        self.mu = mu
        
    def __call__(self, x):
        return 0
        
    def prox(self, x, delta=1.0):
        thresh = delta*self.mu
        thresh *= torch.sqrt(torch.tensor(x.shape[-1]))
        
        ret = torch.clone(x)
        nx = torch.norm(x,p=2,dim=1).view(x.shape[0],1)       
        
        ind = torch.where((nx!=0))[0]
        
        ret[ind] = x[ind] * torch.clamp(1 - torch.clamp(thresh/nx[ind], max=1), min=0)
        return ret
    
        
    def sub_grad(self, x):
        nx = torch.norm(x,p=2,dim=1).view(x.shape[0],1)      
        ind = torch.where((nx!=0))[0]
        ret = torch.clone(x)
        ret[ind] = x[ind]/nx[ind]
        return self.mu * ret
    
# subclass for convolutional kernels
class reg_l1_l2_conv(reg_l1_l2):
    def __init__(self, mu=1.0):
        super().__init__(mu = mu)
        
    def __call__(self, x):
        return super.__call__(x.view(x.shape[0]*x.shape[1],-1))
    
    def prox(self, x, delta=1.0):
        ret = super().prox(x.view(x.shape[0]*x.shape[1],-1), delta)
        return ret.view(x.shape)
    
    def sub_grad(self, x):
        ret = super().sub_grad(x.view(x.shape[0]*x.shape[1],-1))
        return ret.view(x.shape) 
                

        
class reg_l1_l1_l2:        
    def __init__(self, mu=1.0):
        self.mu = mu
        #TODO Add suitable normalization based on layer size
        self.l1 = reg_l1(mu=self.mu)
        self.l1_l2 = reg_l1_l2(mu=self.mu)
        
    def __call__(self, x):
        return 0
        
    def prox(self, x, delta=1.0):
        thresh = delta * self.mu
                
        return self.l1_l2.prox(self.l1.prox(x,thresh), thresh)
    
    def sub_grad(self, x):
        return self.mu * (self.l1.sub_grad(x) + self.l1_l2.sub_grad(x))
