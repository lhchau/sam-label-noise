import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from optimizer import *
import random
import matplotlib.patches as patches
import matplotlib.patches as mpatches


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
    mean_0, cov_0 = [2, 2], [[1, 0], [0, 1]]
    mean_1, cov_1 = [-2, -2], [[1, 0], [0, 1]]
    
    X0 = np.random.multivariate_normal(mean_0, cov_0, n_samples // 2)
    X1 = np.random.multivariate_normal(mean_1, cov_1, n_samples // 2)
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
    
    if noise_rate != 0:
        noise_indices = np.random.choice(n_samples, int(noise_rate * n_samples), replace=False)
        noise_mask = np.zeros(n_samples, dtype=int)
        noise_mask[noise_indices] = 1
        y[noise_indices] = 1 - y[noise_indices]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), noise_mask
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def visualize_dataset(X, y, title, filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0", alpha=0.6, edgecolors='k')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1", alpha=0.6, edgecolors='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title(title)
    plt.savefig(filename, format='pdf')
    
rho = 3
lr = 1
momentum = 0
EPOCHS = 20
INIT = [-8, -2]
opt_name = ['sgd', 'sam', 'saner']

n_samples = 200
batch_size = 200
noise_rate = 0.3

X, y = generate_data(n_samples=n_samples)
train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

X_noise, y_noise, noise_mask = generate_data(n_samples=n_samples, noise_rate=noise_rate)
train_dataset_noise = TensorDataset(X_noise, y_noise)
train_loader_noise = DataLoader(train_dataset_noise, batch_size=batch_size, shuffle=False)

visualize_dataset(X.numpy(), y.numpy(), "Clean Training Dataset", "clean_training_dataset.pdf")
visualize_dataset(X_noise.numpy(), y_noise.numpy(), "Noisy Training Dataset", "noisy_training_dataset.pdf")

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        self.linear.weight.data = torch.tensor([INIT], dtype=torch.float32)
    
    def forward(self, x):
        return self.linear(x).squeeze(1)

def train_model_sgd(train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    trajectory = [model.linear.weight.data.clone().cpu().numpy().flatten()]
    for epoch in range(EPOCHS):
        for idx, (X_batch, y_batch) in enumerate(train_loader):
            # if idx == 0:
            #     print(X_batch, y_batch)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            trajectory.append(model.linear.weight.data.clone().cpu().numpy().flatten())
    return model, trajectory

def train_model_sam(train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = SAM(model.parameters(), rho=rho, lr=lr, weight_decay=0, momentum=momentum)
    trajectory = [model.linear.weight.data.clone().cpu().numpy().flatten()]
    for epoch in range(EPOCHS):
        for idx, (X_batch, y_batch) in enumerate(train_loader):
            # if idx == 0:
            #     print(X_batch, y_batch)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(X_batch), y_batch).backward()
            optimizer.second_step(zero_grad=True)
            trajectory.append(model.linear.weight.data.clone().cpu().numpy().flatten())
    return model, trajectory

def train_model_saner(train_loader, log_groupB=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearModel().to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = SANER(model.parameters(), rho=rho, lr=lr, weight_decay=0, momentum=momentum, group='B', condition=0.1)
    trajectory = [model.linear.weight.data.clone().cpu().numpy().flatten()]
    for epoch in range(EPOCHS):
        cnt = 0
        for idx, (X_batch, y_batch) in enumerate(train_loader):
            # if idx == 0:
            #     print(X_batch, y_batch)
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
            cnt += optimizer.num_groupB
            trajectory.append(model.linear.weight.data.clone().cpu().numpy().flatten())
            if log_groupB:
                print(f'Number Group B: {optimizer.num_groupB}.\t Number Group C: {optimizer.num_groupC}\n')
        print(f'Epoch {epoch}. % Group B {cnt/(idx+1)}')
    return model, trajectory

def train_model(train_loader, optimizer='sgd', log_groupB=False):
    if optimizer == 'sgd':
        return train_model_sgd(train_loader)
    elif optimizer == 'sam':
        return train_model_sam(train_loader)
    elif optimizer == 'saner':
        return train_model_saner(train_loader, log_groupB)

def visualize_loss_landscape_combine(model, X, y, trajectory_sgd, trajectory_sam, trajectory_saner, noisy_final, clean_final, title, filename):
    w1_range = torch.linspace(-9, 2.5, 100)
    w2_range = torch.linspace(-2.5, 2.5, 100)
    
    W1, W2 = torch.meshgrid(w1_range, w2_range, indexing="ij")
    loss_values = torch.zeros_like(W1)
    criterion = nn.BCEWithLogitsLoss()
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            model_noisy.linear.weight.data = torch.tensor([[W1[i, j], W2[i, j]]], dtype=torch.float32)
            outputs = model_noisy(X_noise).detach()
            loss_values[i, j] = criterion(outputs, y_noise)
    mask = loss_values > 1
    overlay = np.ma.masked_where(mask, loss_values)
    
    W1, W2 = torch.meshgrid(w1_range, w2_range, indexing="ij")
    loss_values = torch.zeros_like(W1)
    criterion = nn.BCEWithLogitsLoss()
    
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            model.linear.weight.data = torch.tensor([[W1[i, j], W2[i, j]]], dtype=torch.float32)
            outputs = model(X).detach()
            loss_values[i, j] = criterion(outputs, y)
    
    fontsize = 18
    number_size = 16
    legend_size = 12.5
    icon_size = 300
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.contourf(W1.numpy(), W2.numpy(), loss_values.numpy(), levels=50) # , cmap='coolwarm'
    plt.colorbar()
    plt.contourf(W1, W2, overlay, levels=20, cmap='Reds', alpha=0.5)
    overlay_patch = mpatches.Patch(color='red', alpha=0.5, label="Overfitting Region")
    # plt.colorbar()
    plt.xlabel("Weight 1", fontsize=fontsize)
    plt.ylabel("Weight 2", fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=number_size)
    
    bbox_x, bbox_y = -8, -2  # Bottom-left corner
    bbox_width, bbox_height = 4.5, 4  # Width and height

    rect = patches.Rectangle((bbox_x, bbox_y), bbox_width, bbox_height, 
                            linewidth=3, edgecolor='green', facecolor='none', linestyle="solid")
    ax.add_patch(rect)
    
    # plt.scatter(noisy_final[0], noisy_final[1], marker='o', color='yellow', label='Noisy Minima', s=100)
    # plt.scatter(clean_final[0], clean_final[1], marker='s', color='yellow', label='Overfitting to Noisy Label Minima', s=300)
    
    trajectory_sgd = np.array(trajectory_sgd)
    plt.plot(trajectory_sgd[:, 0], trajectory_sgd[:, 1], marker='o', markersize=2, color='red', linestyle='dashed')
    plt.scatter(trajectory_sgd[-1, 0], trajectory_sgd[-1, 1], marker='*', color='red', label='Final SGD', s=icon_size)
    
    trajectory_sam = np.array(trajectory_sam)
    plt.plot(trajectory_sam[:, 0], trajectory_sam[:, 1], marker='o', markersize=2, color='cyan', linestyle='dashed')
    plt.scatter(trajectory_sam[-1, 0], trajectory_sam[-1, 1], marker='*', color='cyan', label='Final SAM', s=icon_size)
    
    # trajectory_saner = np.array(trajectory_saner)
    # plt.plot(trajectory_saner[:, 0], trajectory_saner[:, 1], marker='o', markersize=2, color='blue', linestyle='dashed')
    # plt.scatter(trajectory_saner[-1, 0], trajectory_saner[-1, 1], marker='*', color='blue', label='Final SANER', s=150)

    plt.scatter(trajectory_sgd[0, 0], trajectory_sgd[0, 1], marker='o', color='orange', label='Initial Weight', s=icon_size)
    
    overlay_legend = plt.legend(handles=[overlay_patch], loc="lower right", fontsize=legend_size)
    plt.gca().add_artist(overlay_legend)  # Keep the first legend while adding the second
    plt.legend(fontsize=legend_size)

    
    plt.savefig(filename, format='pdf')
    
temp_epochs = EPOCHS
temp_lr = lr
temp_INIT = INIT
INIT = [5, 5]
EPOCHS = 50
lr = 1
X_all_noise, y_all_noise = X_noise[noise_mask == 1], y_noise[noise_mask == 1]
train_dataset_all_noise = TensorDataset(X_all_noise, y_all_noise)
train_loader_all_noise = DataLoader(train_dataset_all_noise, batch_size=len(X_all_noise), shuffle=False)
_, trajectory_all_noise = train_model(train_loader_all_noise, 'sgd')

X_all_clean, y_all_clean = X_noise, y_noise
train_dataset_all_clean = TensorDataset(X_all_clean, y_all_clean)
train_loader_all_clean = DataLoader(train_dataset_all_clean, batch_size=len(X_all_clean), shuffle=False)
_, trajectory_all_clean = train_model(train_loader_all_clean, 'sgd')

EPOCHS = temp_epochs
lr = temp_lr
INIT = temp_INIT
traj_list = []
for opt in opt_name:
    model_noisy, trajectory_noisy = train_model(train_loader_noise, opt, log_groupB=False)
    traj_list.append(trajectory_noisy)

visualize_loss_landscape_combine(model_noisy, X_noise[noise_mask == 0], y_noise[noise_mask == 0], traj_list[0], traj_list[1], traj_list[2], trajectory_all_noise[-1], trajectory_all_clean[-1], "Loss Landscape of Training Data with Clean Labels", f"loss_landscape_combine_{opt}.pdf")
