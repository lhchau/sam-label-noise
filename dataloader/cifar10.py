import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from .cutout import Cutout

def apply_label_noise(labels, noise, num_classes):
    """Applies label noise to the clean labels in the proportion specified in :param noise_level.
    This implementation is due to Chiyuan Zhang, see:
    https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py.
    """
    np.random.seed(1996)
    flip_label = np.random.rand(len(labels)) <= noise
    random_labels = np.random.choice(num_classes, flip_label.sum())

    # For the labels where flip_label is True, replace the labels with random_labels.
    labels[flip_label] = random_labels
    return labels


def get_cifar100(
    batch_size=128,
    num_workers=4,
    noise=0.25):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])        
        # Cutout()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
    ])
    labels = apply_label_noise(labels=torch.load("./dataloader/CIFAR-10_human.pt")["clean_label"], noise=noise, num_classes=10)
    
    data_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    data_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    new_data_train = []
    for i in range(len(data_train)):
        new_data_train.append((data_train[i][0], labels[i]))
    
    train_dataloader = torch.utils.data.DataLoader(new_data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_dataloader, test_dataloader, len(data_test.classes)