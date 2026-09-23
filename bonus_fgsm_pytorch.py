"""
BONUS EXTENSION: True FGSM Attack with PyTorch CNN
====================================================
This script trains a small CNN on MNIST, then applies the
real Fast Gradient Sign Method (FGSM) using backpropagated gradients.

Requirements:
    pip install torch torchvision matplotlib numpy

Run after the main script to compare sklearn vs. gradient-based attacks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 1. SIMPLE CNN
# ─────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ─────────────────────────────────────────────
# 2. DATA
# ─────────────────────────────────────────────

def get_loaders(batch_size=128):
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST("./data", train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dl, test_dl


# ─────────────────────────────────────────────
# 3. TRAINING
# ─────────────────────────────────────────────

def train_cnn(epochs=5):
    train_dl, test_dl = get_loaders()
    model = SimpleCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in test_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                correct += (model(X).argmax(1) == y).sum().item()
                total   += len(y)
        print(f"  Epoch {epoch}/{epochs}  loss={total_loss/len(train_dl):.4f}  "
              f"test_acc={correct/total:.4f}")

    return model, test_dl


# ─────────────────────────────────────────────
# 4. TRUE FGSM ATTACK
# ─────────────────────────────────────────────

def fgsm_attack(model, X, y, epsilon):
    """Classic FGSM: x_adv = x + ε * sign(∇_x L(f(x), y))"""
    X = X.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(X), y)
    model.zero_grad()
    loss.backward()
    return torch.clamp(X + epsilon * X.grad.sign(), 0, 1).detach()


# ─────────────────────────────────────────────
# 5. FRAGILITY CURVE (CNN + true FGSM)
# ─────────────────────────────────────────────

def cnn_fragility_curve(model, test_dl, epsilons=None):
    if epsilons is None:
        epsilons = np.linspace(0, 0.5, 21)

    model.eval()
    accuracies = []

    for eps in epsilons:
        correct = total = 0
        for X, y in test_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            X_adv = fgsm_attack(model, X, y, eps)
            correct += (model(X_adv).argmax(1) == y).sum().item()
            total   += len(y)
        accuracies.append(correct / total)
        print(f"  ε={eps:.3f}  acc={correct/total:.4f}")

    return epsilons, accuracies


# ─────────────────────────────────────────────
# 6. VISUAL AUDIT (CNN)
# ─────────────────────────────────────────────

def cnn_visual_audit(model, test_dl, epsilon=0.2, n=6, save_path="cnn_visual_audit.png"):
    model.eval()
    examples = []

    for X, y in test_dl:
        X, y = X.to(DEVICE), y.to(DEVICE)
        X_adv = fgsm_attack(model, X, y, epsilon)

        orig_pred = model(X).argmax(1)
        adv_pred  = model(X_adv).argmax(1)

        tricked = (orig_pred == y) & (adv_pred != y)
        idxs = tricked.nonzero(as_tuple=True)[0]

        for i in idxs:
            examples.append((
                X[i].cpu().squeeze().numpy(),
                X_adv[i].cpu().squeeze().numpy(),
                y[i].item(), orig_pred[i].item(), adv_pred[i].item()
            ))
            if len(examples) >= n:
                break
        if len(examples) >= n:
            break

    fig, axes = plt.subplots(n, 2, figsize=(6, n * 2))
    fig.patch.set_facecolor("#111")
    fig.suptitle(f"CNN FGSM Audit  |  ε={epsilon}", color="white", fontsize=13, y=1.01)

    for i, (clean, adv, gt, op, ap) in enumerate(examples):
        axes[i, 0].imshow(clean, cmap="gray")
        axes[i, 0].set_title(f"Clean  gt={gt} pred={op}", color="#76ff03", fontsize=8)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(adv, cmap="inferno")
        axes[i, 1].set_title(f"Adversarial pred={ap} ❌", color="#ff1744", fontsize=8)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#111")
    plt.show()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Training CNN...")
    model, test_dl = train_cnn(epochs=5)

    print("\nComputing FGSM Fragility Curve...")
    epsilons = np.linspace(0, 0.5, 21)
    eps_vals, accs = cnn_fragility_curve(model, test_dl, epsilons)

    plt.figure(figsize=(8, 4))
    plt.plot(eps_vals, accs, "o-", color="cyan", lw=2)
    plt.axhline(accs[0], color="lime", ls="--", label=f"Baseline {accs[0]:.2%}")
    plt.axhline(accs[0] * 0.5, color="red", ls=":", label="50% Drop")
    plt.xlabel("ε"); plt.ylabel("Accuracy")
    plt.title("CNN FGSM Fragility Curve")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("cnn_fragility_curve.png", dpi=150)
    plt.show()

    print("\nGenerating CNN Visual Audit...")
    cnn_visual_audit(model, test_dl, epsilon=0.2)
