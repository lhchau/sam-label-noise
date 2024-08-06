import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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


def get_cifar100(label_type, noise, data_transforms):
    """Gets the CIFAR-100 dataset with the labels selected by the user."""

    if label_type != "blue":
        labels = torch.load("./data/CIFAR-100_human.pt")[f"{label_type}_label"]
    else:
        labels = apply_label_noise(labels=torch.load("./data/CIFAR-100_human.pt")["clean_label"],
                                   noise=noise,
                                   num_classes=100)
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=data_transforms)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=data_transforms)
    classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
               'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly',
               'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
               'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
               'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl',
               'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard',
               'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
               'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
               'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
               'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
               'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
               'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
               'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf',
               'woman', 'worm')

    return labels, train_set, test_set, classes


def get_cifar10(label_type, noise, data_transforms):
    """Gets the CIFAR-10 dataset with the labels selected by the user.
    There are two types of noise: red and blue, corresponding to "natural" and "synthetic" noise.
    Red noise has 2 levels: aggregate and worse. Blue has any value [0, 1].
    """

    if label_type != "blue":
        # Load whichever type of red noise labels the user specified: clean, aggre, worse.
        labels = torch.load("./data/CIFAR-10_human.pt")[f"{label_type}_label"]
    else:
        labels = apply_label_noise(labels=torch.load("./data/CIFAR-10_human.pt")["clean_label"],
                                   noise=noise,
                                   num_classes=10)

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return labels, train_set, test_set, classes


class CIFAR:
    def __init__(self, which_dataset="cifar10", batch_size=128, label_type="clean", noise=0.0, threads=0):
        data_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if which_dataset == "cifar10":
            labels, train_set, test_set, classes = get_cifar10(label_type, noise, data_transforms)
        else:
            labels, train_set, test_set, classes = get_cifar100(label_type, noise, data_transforms)

        # Replace the labels in the training set
        # with the labels from the labels array.
        new_training_set = []

        for i in range(len(train_set)):
            new_training_set.append((train_set[i][0], labels[i]))

        self.train = torch.utils.data.DataLoader(new_training_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)
        self.classes = classes
