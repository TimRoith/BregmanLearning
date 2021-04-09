import torch
import torch.nn.functional as F
import models

class Conf:
    def __init__(self, **kwargs):

        # dataset
        self.data_set = kwargs.get('data_set', "MNIST")
        self.data_set_mean = 0.0
        self.data_set_std = 1.0
        self.data_file = kwargs.get('data_file', "data")
        self.train_split = kwargs.get('train_split', 1.0)
        self.im_shape = None
        self.x_min = 0.0
        self.x_max = 1.0
        
        # CUDA settings
        self.use_cuda = kwargs.get('use_cuda', False)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.num_workers = kwargs.get('num_workers', 0)
        
        
        # Loss function and norm
        self.loss = kwargs.get('loss', F.cross_entropy)
        
        # -----------------------------
        self.goal_acc = kwargs.get('goal_accuracy', 0.0)
 
        # specification for Training
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 128)
        self.lr = kwargs.get('lr', 0.1)
            
            
# -----------------------------
# Examples
# -----------------------------

# -----------------------------
# no regularization
# -----------------------------
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

def fashion_mnist_example(data_file, use_cuda=False, num_workers=None):
    if use_cuda and num_workers is None:
        num_workers = 4
    else:
        num_workers = 0
    
    conf_args = {'lamda':0.0,'data_file':data_file, 'use_cuda':use_cuda, 'train_split':0.9, 'num_workers':num_workers,
                 'regularization': "none", 'activation_function':"sigmoid"}

    # get configuration
    conf = Conf(**conf_args)
    
    return conf
            
