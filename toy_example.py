import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from optimizer import *
import random


def initialize(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


initialize(42)


def generate_data(n_samples=6, noise_rate=0):
    mean_0, cov_0 = [1, 1], [[1, 0.25], [0.25, 1]]
    mean_1, cov_1 = [-1, -1], [[1, -0.25], [-0.25, 1]]

    X0 = np.random.multivariate_normal(mean_0, cov_0, n_samples // 2)
    X1 = np.random.multivariate_normal(mean_1, cov_1, n_samples // 2)
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    if noise_rate != 0:
        noise_indices = np.random.choice(
            n_samples, int(noise_rate * n_samples), replace=False)
        y[noise_indices] = 1 - y[noise_indices]

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def visualize_dataset(X, y, title, filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1],
                label="Class 0", alpha=0.6, edgecolors='k')
    plt.scatter(X[y == 1, 0], X[y == 1, 1],
                label="Class 1", alpha=0.6, edgecolors='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title(title)
    plt.savefig(filename, format='pdf')


batch_size = 100
n_samples = 100
X, y = generate_data(n_samples=n_samples)
train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

X_noise, y_noise = generate_data(n_samples=n_samples, noise_rate=0.25)
train_dataset_noise = TensorDataset(X_noise, y_noise)
train_loader_noise = DataLoader(
    train_dataset_noise, batch_size=batch_size, shuffle=True)

visualize_dataset(X.numpy(), y.numpy(),
                  "Clean Training Dataset", "clean_training_dataset.pdf")
visualize_dataset(X_noise.numpy(), y_noise.numpy(),
                  "Noisy Training Dataset", "noisy_training_dataset.pdf")


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        self.linear.weight.data = torch.tensor([[1, -5]], dtype=torch.float32)

    def forward(self, x):
        return self.linear(x).squeeze(1)


rho = 2
lr = 0.1
EPOCHS = 100
opt_name = ['sgd', 'sam', 'saner']


def train_model_sgd(train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    trajectory = [model.linear.weight.data.clone().cpu().numpy().flatten()]
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            trajectory.append(
                model.linear.weight.data.clone().cpu().numpy().flatten())
    return model, trajectory


def train_model_sam(train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = SAM(model.parameters(), rho=rho,
                    lr=lr, weight_decay=0, momentum=0)
    trajectory = [model.linear.weight.data.clone().cpu().numpy().flatten()]
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(X_batch), y_batch).backward()
            optimizer.second_step(zero_grad=True)
            trajectory.append(
                model.linear.weight.data.clone().cpu().numpy().flatten())
    return model, trajectory


def train_model_saner(train_loader, log_groupB=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearModel().to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = SANER(model.parameters(), rho=rho, lr=lr,
                      weight_decay=0, momentum=0, group='B', condition=0)
    trajectory = [model.linear.weight.data.clone().cpu().numpy().flatten()]
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            torch.mean(loss).backward()
            optimizer.first_step(zero_grad=True)
            torch.mean(criterion(model(X_batch), y_batch)).backward()
            optimizer.second_step(zero_grad=False)
            # if optimizer.num_groupB == 1:
            #     breakpoint()
            trajectory.append(
                model.linear.weight.data.clone().cpu().numpy().flatten())
            if log_groupB:
                print(
                    f'Number Group B: {optimizer.num_groupB}.\t Number Group C: {optimizer.num_groupC}\n')
    return model, trajectory


def train_model(train_loader, optimizer='sgd', log_groupB=False):
    if optimizer == 'sgd':
        return train_model_sgd(train_loader)
    elif optimizer == 'sam':
        return train_model_sam(train_loader)
    elif optimizer == 'saner':
        return train_model_saner(train_loader, log_groupB)


def visualize_loss_landscape(model, X, y, trajectory, title, filename):
    w1_range = torch.linspace(-6, 6, 50)
    w2_range = torch.linspace(-6, 6, 50)
    W1, W2 = torch.meshgrid(w1_range, w2_range, indexing="ij")

    loss_values = torch.zeros_like(W1)
    criterion = nn.BCEWithLogitsLoss()

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            model.linear.weight.data = torch.tensor(
                [[W1[i, j], W2[i, j]]], dtype=torch.float32)
            outputs = model(X).detach()
            loss_values[i, j] = criterion(outputs, y)

    plt.figure(figsize=(8, 6))
    plt.contourf(W1.numpy(), W2.numpy(), loss_values.numpy(),
                 levels=50, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel("Weight 1")
    plt.ylabel("Weight 2")
    plt.title(title)

    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o',
             markersize=2, color='black', linestyle='dashed')

    plt.scatter(trajectory[0, 0], trajectory[0, 1],
                marker='s', color='blue', label='Init', s=100)
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1],
                marker='*', color='red', label='Final', s=150)
    plt.legend()

    plt.savefig(filename, format='pdf')


for opt in opt_name:
    model_clean, trajectory_clean = train_model(train_loader, opt)
    model_noisy, trajectory_noisy = train_model(
        train_loader_noise, opt, log_groupB=True)
    # visualize_loss_landscape(model_clean, X, y, trajectory_clean, "Loss Landscape (Clean Data)", f"loss_landscape_clean_{opt}.pdf")
    visualize_loss_landscape(model_noisy, X_noise, y_noise, trajectory_noisy,
                             "Loss Landscape (Noisy Data)", f"loss_landscape_noisy_{opt}.pdf")
