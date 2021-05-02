from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

import os


def get_data_set(conf, test_size=1):
    train, valid, test, train_loader, valid_loader, test_loader = [None] * 6
    if conf.data_set == "MNIST":
        train, test = get_mnist(conf)
    elif conf.data_set == "Fashion-MNIST":
        train, test = get_fashion_mnist(conf)
    elif conf.data_set == "CIFAR10":
        train, test = get_cifar10(conf)
    else:
        raise ValueError("Dataset:" + conf.data_set + " not defined")
    
    # get loaders fom datasets
    train_loader, valid_loader, test_loader = train_valid_test_split(conf, train, test)

    return train_loader, valid_loader, test_loader

# Get the MNIST dataset and apply transformations
def get_mnist(conf):
    transform = transforms.Compose([transforms.ToTensor()])
    
    # train and test set
    train = datasets.MNIST(conf.data_file, train=True, download=False, transform=transform)
    test = datasets.MNIST(conf.data_file, train=False, download=False, transform=transform)
    
    return train, test

# Get the Fashion-MNIST dataset and apply transformations
def get_fashion_mnist(conf):
    transform = transforms.Compose([transforms.ToTensor()])
    
    # train and test set
    train = datasets.FashionMNIST(conf.data_file, train=True, download=False,transform=transform)
    test = datasets.FashionMNIST(conf.data_file, train=False, download=False, transform=transform)
    
    return train, test


# Get the cifar dataset and apply transformations
def get_cifar10(conf):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

    transform_test = transforms.Compose([transforms.ToTensor()])
    
   # train and test set
    train = datasets.CIFAR10(conf.data_file, train=True, download=False, transform=transform_train)
    test = datasets.CIFAR10(conf.data_file, train=False, download=False, transform=transform_test)
    
    return train, test



def train_valid_test_split(conf, train, test):
    total_count = len(train)
    train_count = int(conf.train_split * total_count)
    val_count = total_count - train_count
    if val_count > 0:
        train, val = torch.utils.data.random_split(train, [train_count, val_count],generator=torch.Generator().manual_seed(42))
        valid_loader = DataLoader(val, batch_size=128, shuffle=True, pin_memory=False)
    else:
        valid_loader = None

    train_loader = DataLoader(train, batch_size=conf.batch_size, shuffle=True, pin_memory=True, num_workers=conf.num_workers)
    test_loader = DataLoader(test, batch_size=1000, shuffle=False, pin_memory=True, num_workers=conf.num_workers)

    return train_loader, valid_loader, test_loader


# set imshape, mean and std for the dataset
def data_set_info(data_set, device=None):
    if data_set == "MNIST":
        im_shape = [1,28,28]
        data_set_mean = 0.1307
        data_set_std = 0.3081
    elif data_set == "Fashion-MNIST":
        im_shape = [1,28,28]
        data_set_mean = 0.5
        data_set_std = 0.5
    elif data_set == "CIFAR10":
        im_shape = [3,32,32]
        data_set_mean = torch.tensor([0.4914, 0.4822, 0.4465],device=device)
        data_set_std = torch.tensor([0.2023, 0.1994, 0.2010],device=device)

    else:
        raise ValueError("Dataset:" + data_set + " not defined")
        
    return im_shape, data_set_mean, data_set_std
    
