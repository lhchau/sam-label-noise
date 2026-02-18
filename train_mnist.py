import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from optimizer import SAM, SANER
from tqdm import tqdm
from utils import *

initialize(42)
# -----------------------------
# 1. Dataset: Binary MNIST
# -----------------------------
def get_binary_mnist(label_a=3, label_b=8, noise_ratio=0.4, samples_per_class=500):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor in [0,1]
        # transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with mean and std
        transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 to 784
    ])

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    def filter_and_map(dataset):
        idx_a = [i for i, t in enumerate(dataset.targets) if t == label_a]
        idx_b = [i for i, t in enumerate(dataset.targets) if t == label_b]
        
        idx_a = np.random.choice(idx_a, samples_per_class, replace=False)
        idx_b = np.random.choice(idx_b, samples_per_class, replace=False)

        selected_indices = np.concatenate([idx_a, idx_b])
        np.random.shuffle(selected_indices)

        dataset.data = dataset.data[selected_indices]
        dataset.targets = dataset.targets[selected_indices]
        dataset.targets = (dataset.targets == label_b).long()  # label_b -> 1, label_a -> 0

        return dataset

    train_set = filter_and_map(train_set)

    num_noisy = int(noise_ratio * len(train_set))
    noisy_indices = np.random.choice(len(train_set), num_noisy, replace=False)
    train_set.targets[noisy_indices] = 1 - train_set.targets[noisy_indices]

    clean_set = Subset(train_set, [i for i in range(len(train_set)) if i not in noisy_indices and train_set.targets[i] == 0])
    noisy_set = Subset(train_set, [i for i in range(len(train_set)) if i in noisy_indices and train_set.targets[i] == 1])

    return train_set, clean_set, noisy_set

# -----------------------------
# 2. Model: Overfitting MLP
# -----------------------------
class OverfitMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, 1)
            
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# -----------------------------
# 3. Train function
# -----------------------------
def train_model_sgd(model, train_loader, epochs=100, x_c=None, x_n=None, lr=0.1, rho=0.05):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    param_snapshots = []
    z_c = []
    z_n = []
    for epoch in tqdm(range(epochs)):
        with torch.no_grad():
            if x_c is not None and x_n is not None:
                z_c.append(np.round(model(x_c).mean().item(), 2).item())
                z_n.append(np.round(model(x_n).mean().item(), 2).item())
        model.train()
        total_loss, correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += ((torch.sigmoid(out) > 0.5).long() == y.long()).sum().item()
        acc = correct / len(train_loader.dataset)
        param_snapshots.append(get_param_vector(model))
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader.dataset):.4f}, Acc={acc:.4f}")
        # scheduler.step()
    return model, param_snapshots, z_c, z_n

def train_model_sam(model, train_loader, epochs=100, x_c=None, x_n=None, lr=0.1, rho=0.05):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = SAM(model.parameters(), lr=lr, momentum=0, weight_decay=0, rho=rho)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    param_snapshots = []
    z_c = []
    z_n = []
    for epoch in tqdm(range(epochs)):
        with torch.no_grad():
            if x_c is not None and x_n is not None:
                z_c.append(np.round(model(x_c).mean().item(), 2).item())
                z_n.append(np.round(model(x_n).mean().item(), 2).item())
        model.train()
        total_loss, correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(x), y).backward()
            optimizer.second_step(zero_grad=True)
            total_loss += loss.item() * x.size(0)
            correct += ((torch.sigmoid(out) > 0.5).long() == y.long()).sum().item()
        acc = correct / len(train_loader.dataset)
        param_snapshots.append(get_param_vector(model))
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader.dataset):.4f}, Acc={acc:.4f}")
        # scheduler.step()
    return model, param_snapshots, z_c, z_n

def train_model_saner(model, train_loader, epochs=100, x_c=None, x_n=None, lr=0.1, rho=0.05):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = SANER(model.parameters(), lr=lr, momentum=0, weight_decay=0, rho=rho)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    param_snapshots = []
    z_c = []
    z_n = []
    for epoch in tqdm(range(epochs)):
        with torch.no_grad():
            if x_c is not None and x_n is not None:
                z_c.append(np.round(model(x_c).mean().item(), 2).item())
                z_n.append(np.round(model(x_n).mean().item(), 2).item())
        model.train()
        total_loss, correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(x), y).backward()
            optimizer.second_step(zero_grad=True)
            total_loss += loss.item() * x.size(0)
            correct += ((torch.sigmoid(out) > 0.5).long() == y.long()).sum().item()
        acc = correct / len(train_loader.dataset)
        param_snapshots.append(get_param_vector(model))
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader.dataset):.4f}, Acc={acc:.4f}")
        # scheduler.step()
    return model, param_snapshots, z_c, z_n

# -----------------------------
# 4. Loss landscape visualization
# -----------------------------
def get_param_vector(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_param_vector(model, vec):
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.data = vec[pointer:pointer+numel].view_as(p).clone()
        pointer += numel

def loss_landscape_2d(model, dataloader, base_params, param_snapshots, range_val=1.0, steps=15):
    # Random directions
    dim = base_params.size(0)
    d1 = torch.randn(dim).to(device)
    d2 = torch.randn(dim).to(device)
    d1 /= torch.norm(d1)
    d2 /= torch.norm(d2)
    
    coords = []
    for w in param_snapshots:
        dx = torch.dot(w - base_params, d1)
        dy = torch.dot(w - base_params, d2)
        coords.append((dx.item(), dy.item()))

    loss_surface = np.zeros((steps, steps))

    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    for i, alpha in tqdm(enumerate(np.linspace(-range_val, range_val, steps))):
        for j, beta in tqdm(enumerate(np.linspace(-range_val, range_val, steps))):
            vec = base_params + alpha * d1 + beta * d2
            set_param_vector(model, vec)
            with torch.no_grad():
                losses = []
                for x, y in dataloader:
                    x, y = x.to(device), y.float().to(device)
                    out = model(x)
                    loss = criterion(out, y)
                    losses.append(loss.item())
            loss_surface[i, j] = np.mean(losses)

    return loss_surface, coords

def plot_landscape(surface, coords, range_val=1.0):
    plt.figure(figsize=(6, 5))
    extent = [-range_val, range_val, -range_val, range_val]
    plt.imshow(surface, origin='lower', extent=extent, cmap='viridis')
    trajectory = np.array(coords)
    plt.plot(trajectory[:,0], trajectory[:,1], marker='o', color='red', linewidth=2)
    plt.colorbar(label='Loss')
    plt.title("2D Loss Landscape")
    plt.xlabel("Direction d1")
    plt.ylabel("Direction d2")
    plt.tight_layout()
    plt.savefig("toy_example/loss_landscape.png", bbox_inches='tight', dpi=300)
    plt.close()

# c) Plot decision boundary
def plot_decision_boundary_trained_model(model, pca, x_2d, y_np):
    model.eval()
    x_min, x_max = x_2d[:, 0].min() - 1, x_2d[:, 0].max() + 1
    y_min, y_max = x_2d[:, 1].min() - 1, x_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_784d = pca.inverse_transform(grid_2d)
    grid_tensor = torch.tensor(grid_784d, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(grid_tensor).cpu().numpy().reshape(xx.shape)
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid

    plt.contourf(xx, yy, probs, levels=50, cmap='coolwarm', alpha=0.6)
    plt.scatter(x_2d[:, 0], x_2d[:, 1], c=y_np, cmap='coolwarm', edgecolors='k')
    plt.title("Decision Boundary in PCA Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("toy_example/decision_boundary.png", bbox_inches='tight', dpi=300)
    plt.close()

# -----------------------------
# 5. Main
# -----------------------------
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    train_set, clean_set, noisy_set = get_binary_mnist(label_a=0, label_b=1, noise_ratio=0.3, samples_per_class=256)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    
    clean_data_loader = DataLoader(clean_set, batch_size=len(clean_set), shuffle=False)
    noisy_data_loader = DataLoader(noisy_set, batch_size=len(noisy_set), shuffle=False)
    clean_set = next(iter(clean_data_loader))
    noisy_set = next(iter(noisy_data_loader))
    model = OverfitMLP().to(device)
    # trained_model, param_snapshots, z_c, z_n = train_model(model, train_loader, epochs=50, x_c=clean_set[0].to(device), x_n=noisy_set[0].to(device))
    opt = 'diff_saner'
    if opt[-3:] == 'sam':
        trained_model, param_snapshots, z_c, z_n = train_model_sam(model, train_loader, epochs=500, x_c=clean_set[0].to(device), x_n=noisy_set[0].to(device), lr=0.1, rho=0.1)
    elif opt[-5] == 'saner':
        trained_model, param_snapshots, z_c, z_n = train_model_saner(model, train_loader, epochs=500, x_c=clean_set[0].to(device), x_n=noisy_set[0].to(device), lr=0.1, rho=0.1)
    else:
        trained_model, param_snapshots, z_c, z_n = train_model_sgd(model, train_loader, epochs=500, x_c=clean_set[0].to(device), x_n=noisy_set[0].to(device), lr=0.1, rho=0.02)
    np.save(f"toy_example/z_c_{opt}.npy", np.array(z_c))
    np.save(f"toy_example/z_n_{opt}.npy", np.array(z_n))
    
    # Decision boundary visualization
    # x_all, y_all = next(iter(DataLoader(train_set, batch_size=len(train_set))))
    # x_np = x_all.numpy()
    # y_np = y_all.numpy()
    # pca = PCA(n_components=2)
    # x_2d = pca.fit_transform(x_np)
    # plot_decision_boundary_trained_model(trained_model, pca, x_2d, y_np)
    
    # final_params = get_param_vector(trained_model).detach()
    # surface, coords = loss_landscape_2d(trained_model, test_loader, final_params, param_snapshots, range_val=1.0, steps=30)
    # plot_landscape(surface, coords)