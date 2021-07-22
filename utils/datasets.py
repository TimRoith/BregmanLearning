from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import math
import numbers
from torch import nn
from torch.nn import functional as F

import os


def get_data_set(conf):
    train, valid, test, train_loader, valid_loader, test_loader = [None] * 6
    
    if conf.data_set == "MNIST":
        train, test = get_mnist(conf)
        test_batch_size = 1000
    elif conf.data_set == "Fashion-MNIST":
        train, test = get_fashion_mnist(conf)
        test_batch_size = 1000
    elif conf.data_set == "CIFAR10":
        train, test = get_cifar10(conf)
        test_batch_size = 100
    elif conf.data_set == "Encoder-MNIST":
        train, test = get_encoder_mnist(conf)
        test_batch_size = 1000
        pin_memory = True
    else:
        raise ValueError("Dataset:" + conf.data_set + " not defined")
    
    # get loaders fom datasets
    train_loader, valid_loader, test_loader = train_valid_test_split(conf, train, test,test_batch_size = test_batch_size)

    return train_loader, valid_loader, test_loader

# Get the MNIST dataset and apply transformations
def get_mnist(conf):
    transform = transforms.Compose([transforms.ToTensor()])
    
    # train and test set
    train = datasets.MNIST(conf.data_file, train=True, download=conf.download, transform=transform)
    test = datasets.MNIST(conf.data_file, train=False, download=conf.download, transform=transform)
    
    return train, test

# Get the MNIST dataset and apply transformations
def get_encoder_mnist(conf):
    transform_aug = []
    
    if conf.add_blur:
        kernel_size = 5
        sigma=(0.1, 2.0)
        transform_aug.append(transforms.GaussianBlur(kernel_size, sigma=sigma))
    
    transform_aug.append(transforms.ToTensor())
    
    if conf.add_noise:
        transform_aug.append(add_noise(std=0.1))
    
    transform_aug = transforms.Compose(transform_aug)
        
    transform_clean = transforms.Compose([transforms.ToTensor()])
       
    # train and test set
    train = AutoEncodeDataset(datasets.MNIST(conf.data_file, train=True, download=conf.download), transform_aug, transform_clean)
    test = AutoEncodeDataset(datasets.MNIST(conf.data_file, train=False, download=conf.download), transform_aug, transform_clean)
    
    return train, test

# Get the Fashion-MNIST dataset and apply transformations
def get_fashion_mnist(conf):
    transform = transforms.Compose([transforms.ToTensor()])
    
    # train and test set
    train = datasets.FashionMNIST(conf.data_file, train=True, download=conf.download,transform=transform)
    test = datasets.FashionMNIST(conf.data_file, train=False, download=conf.download, transform=transform)
    
    return train, test


# Get the cifar dataset and apply transformations
def get_cifar10(conf):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

    transform_test = transforms.Compose([transforms.ToTensor()])
    
   # train and test set
    train = datasets.CIFAR10(conf.data_file, train=True, download=conf.download, transform=transform_train)
    test = datasets.CIFAR10(conf.data_file, train=False, download=conf.download, transform=transform_test)
    
    return train, test



def train_valid_test_split(conf, train, test, test_batch_size=1000):
    total_count = len(train)
    train_count = int(conf.train_split * total_count)
    val_count = total_count - train_count
    if val_count > 0:
        train, val = torch.utils.data.random_split(train, [train_count, val_count],generator=torch.Generator().manual_seed(42))
        valid_loader = DataLoader(val, batch_size=128, shuffle=True, pin_memory=False)
    else:
        valid_loader = None

    train_loader = DataLoader(train, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    test_loader = DataLoader(test, batch_size=test_batch_size, shuffle=False, pin_memory=conf.pin_memory, num_workers=conf.num_workers)

    return train_loader, valid_loader, test_loader


# set imshape, mean and std for the dataset
def data_set_info(data_set, device=None):
    if data_set == "MNIST" or data_set == "Encoder-MNIST":
        im_shape = [1,28,28]
        data_set_mean = 0.1307
        data_set_std = 0.3081
    elif data_set == "Fashion-MNIST":
        im_shape = [1,28,28]
        data_set_mean = 0.5
        data_set_std = 0.5
    elif data_set == "CIFAR10":
        im_shape = [3,32,32]
        data_set_mean = torch.tensor([0.4914, 0.4822, 0.4465],device=device).view(-1,1,1)
        data_set_std = torch.tensor([0.2023, 0.1994, 0.2010],device=device).view(-1,1,1)

    else:
        raise ValueError("Dataset:" + data_set + " not defined")
        
    return im_shape, data_set_mean, data_set_std
    

class AutoEncodeDataset(Dataset):
    def __init__(self, dataset, transform_aug, transform_clean):
        self.dataset = dataset
        self.transform_aug = transform_aug
        self.transform_clean = transform_clean

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset.__getitem__(idx)

        # ----------------------------
        x_aug = self.transform_aug(x)
        x = self.transform_clean(x)
        return x_aug, x
    
def get_augmented_dataset(conf, dataset, transform):
    X = torch.zeros(size=(len(dataset),*conf.im_shape))
    Y = torch.zeros(size=(len(dataset),*conf.im_shape))
    for i, (x,_) in enumerate(dataset):
        X[i,::] = transform(x.view(1,*x.shape))[0,::]
        Y[i,::] = x
        
    return torch.utils.data.TensorDataset(X, Y)
    
    

class add_noise(object):
    def __init__(self, mean=0., std=1., device = None):
        self.std = std
        self.mean = mean
        self.device = device
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(),device=self.device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
    
    
class GaussianSmoothing(torch.nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        
        # padding
        self.pad = int(0.5*(kernel_size-1)) 
        
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
          
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        
        

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """        
        return self.conv(input, weight=self.weight, groups=self.groups,padding=self.pad)