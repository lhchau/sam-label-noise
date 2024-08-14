import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from .cutout import Cutout

class CIFAR100Noisy(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, noise_rate=0.2):
        super(CIFAR100Noisy, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.noise_rate = noise_rate
        self.noisy_labels = self.targets.copy()  # Copy the original labels

        if self.train:
            self._apply_noise()

    def _apply_noise(self):
        num_samples = len(self.noisy_labels)
        num_noisy = int(self.noise_rate * num_samples)
        noisy_indices = np.random.choice(num_samples, num_noisy, replace=False)

        self.flip_labels = torch.zeros(num_samples, dtype=torch.bool)
        self.flip_labels[noisy_indices] = True

        for idx in noisy_indices:
            current_label = self.noisy_labels[idx]
            new_label = np.random.choice([x for x in range(100) if x != current_label])
            self.noisy_labels[idx] = new_label

    def __getitem__(self, index):
        img, target, flip_label = self.data[index], self.noisy_labels[index], self.flip_labels[index]

        img = Image.fromarray(img)
        
        # Apply the transformations if any
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, flip_label
 
def get_cifar10(
    batch_size=128,
    num_workers=4,
    noise=0.25):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # Cutout()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    data_train = CIFAR100Noisy(root='./data', train=True, download=True, transform=transform_train, noise_rate=noise)
    data_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_dataloader, test_dataloader, len(data_test.classes)