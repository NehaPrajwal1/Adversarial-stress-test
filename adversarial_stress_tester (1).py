"""
PROBLEM 106: THE ADVERSARIAL STRESS-TESTER
============================================
Adversarial robustness analysis on MNIST using RBF-SVM / Random Forest.
Implements: Model Hardening, Perturbation Engine, Fragility Curve, Visual Audit.

Requirements:
    pip install scikit-learn numpy matplotlib openml tqdm
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import openml
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------

def load_mnist(n_train=10000, n_test=2000, random_state=42):
    """Load MNIST-784 from OpenML (dataset ID 554)."""
    print("Loading MNIST from OpenML (ID=554)...")
    dataset = openml.datasets.get_dataset(554)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    X = X.values.astype(np.float32)
    y = y.astype(int).values

    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, y_train = X[:n_train], y[:n_train]
    X_test,  y_test  = X[n_train:n_train + n_test], y[n_train:n_train + n_test]

    X_train = X_train / 255.0
    X_test  = X_test  / 255.0

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# 2. MODEL HARDENING
# ---------------------------------------------------------

def train_rbf_svm(X_train, y_train):
    """Train an RBF-kernel SVM (pipeline with StandardScaler)."""
    print("\nTraining RBF-SVM (this may take a minute)...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale", probability=False))
    ])
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=200):
    """Train a Random Forest classifier."""
    print(f"\nTraining Random Forest ({n_estimators} trees)...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     RandomForestClassifier(n_estimators=n_estimators,
                                          n_jobs=-1, random_state=42))
    ])
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------
# 3. PERTURBATION ENGINE
# ---------------------------------------------------------

def gradient_sign_perturbation(X, epsilon, random_state=0):
    """
    Approximates FGSM using random signed noise (no gradients needed).
    Works on any sklearn model.
    """
    rng = np.random.default_rng(random_state)
    noise = rng.choice([-1.0, 1.0], size=X.shape).astype(np.float32)
    X_adv = X + epsilon * noise
    return np.clip(X_adv, 0.0, 1.0)


def targeted_sign_perturbation(X, model, epsilon, n_directions=5, random_state=0):
    """
    Query-based attack: tries n_directions random sign vectors,
    keeps the one that causes the most misclassifications.
    """
    rng = np.random.default_rng(random_state)
    best_X_adv = X.copy()
    best_errors = 0
    orig_preds = model.predict(X)

    for _ in range(n_directions):
        noise = rng.choice([-1.0, 1.0], size=X.shape).astype(np.float32)
        X_adv = np.clip(X + epsilon * noise, 0.0, 1.0)
        preds = model.predict(X_adv)
        errors = np.sum(preds != orig_preds)
        if errors > best_errors:
            best_errors = errors
            best_X_adv = X_adv

    return best_X_adv


# ---------------------------------------------------------
# 4. FRAGILITY CURVE
# ---------------------------------------------------------

def compute_fragility_curve(model, X_test, y_test,
                             epsilons=None, attack="random"):
    """
    Sweep epsilon values, compute accuracy at each level.
    Returns (epsilons, accuracies, baseline_acc, breaking_eps).
    """
    if epsilons is None:
        epsilons = np.linspace(0, 0.5, 21)

    baseline_acc = accuracy_score(y_test, model.predict(X_test))
    target_acc   = baseline_acc * 0.50
    breaking_eps = None
    accuracies   = []

    print(f"\nComputing fragility curve ({len(epsilons)} epsilon values)...")
    print(f"Baseline accuracy: {baseline_acc:.4f}  |  50% drop target: {target_acc:.4f}")

    for eps in epsilons:
        if attack == "random":
            X_adv = gradient_sign_perturbation(X_test, eps)
        else:
            X_adv = targeted_sign_perturbation(X_test, model, eps, n_directions=3)

        acc = accuracy_score(y_test, model.predict(X_adv))
        accuracies.append(acc)

        if breaking_eps is None and acc <= target_acc:
            breaking_eps = eps
            print(f"Breaking Point found at epsilon = {eps:.4f}  (acc = {acc:.4f})")

    if breaking_eps is None:
        print("Model did not reach 50% accuracy drop within tested epsilon range.")

    return epsilons, accuracies, baseline_acc, breaking_eps


def plot_fragility_curve(epsilons, accuracies, baseline_acc, breaking_eps,
                         model_name="Model", save_path="fragility_curve.png"):

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    ax.grid(True, color="#1e1e1e", lw=0.8, zorder=0)

    # Main curve
    ax.plot(epsilons, accuracies, color="#00e5ff", lw=2.5,
            marker="o", markersize=6, markerfacecolor="white",
            markeredgecolor="#00e5ff", markeredgewidth=1.5,
            label="Adversarial Accuracy", zorder=3)

    # Baseline accuracy
    ax.axhline(baseline_acc, color="#76ff03", lw=1.8, ls="--",
               label=f"Clean Baseline  {baseline_acc:.2%}", zorder=2)

    # 50% drop threshold
    threshold = baseline_acc * 0.5
    ax.axhline(threshold, color="#ff1744", lw=1.5, ls=":",
               label=f"50% Drop Threshold  ({threshold:.2%})", zorder=2)

    # Breaking point vertical line + annotation
    if breaking_eps is not None:
        ax.axvline(breaking_eps, color="#ff9100", lw=2, ls="--",
                   label=f"Breaking Point  epsilon = {breaking_eps:.3f}", zorder=2)

        ax.scatter([breaking_eps], [threshold],
                   s=160, color="#ff9100", zorder=5,
                   edgecolors="white", lw=1.5)

        ax.annotate(
            f"  Breaking Point\n  epsilon = {breaking_eps:.3f}",
            xy=(breaking_eps, threshold),
            xytext=(breaking_eps + 0.03, threshold + 0.06),
            color="#ff9100", fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#ff9100", lw=1.5),
            zorder=6
        )

    # Shade danger zone
    accuracies_arr = np.array(accuracies)
    ax.fill_between(epsilons, 0, accuracies_arr,
                    where=(accuracies_arr <= threshold),
                    color="#ff1744", alpha=0.08)

    ax.set_xlabel("Epsilon - Perturbation Strength", color="#cccccc", fontsize=12)
    ax.set_ylabel("Accuracy Score", color="#cccccc", fontsize=12)
    ax.set_title(f"Fragility Curve - {model_name}",
                 color="white", fontsize=14, pad=14, fontweight="bold")

    ax.tick_params(colors="#aaaaaa", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.set_xlim(epsilons[0] - 0.01, epsilons[-1] + 0.01)
    ax.set_ylim(0.25, baseline_acc + 0.08)

    ax.legend(facecolor="#1a1a1a", labelcolor="white",
              framealpha=0.9, fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved -> {save_path}")


# ---------------------------------------------------------
# 5. VISUAL AUDIT
# ---------------------------------------------------------

def visual_audit(model, X_test, y_test, epsilon=0.15,
                 n_pairs=6, save_path="visual_audit.png"):
    """
    Display n_pairs of (Clean | Adversarial) digits with model predictions.
    Focuses on samples where the model is tricked.
    """
    X_adv = gradient_sign_perturbation(X_test, epsilon)
    orig_preds = model.predict(X_test)
    adv_preds  = model.predict(X_adv)

    tricked_idx = np.where(
        (orig_preds == y_test) & (adv_preds != y_test)
    )[0]

    if len(tricked_idx) == 0:
        print(f"No misclassifications at epsilon={epsilon:.2f}. Using random samples.")
        tricked_idx = np.arange(min(n_pairs, len(X_test)))

    selected = tricked_idx[:n_pairs]

    fig = plt.figure(figsize=(13, 2.8 * n_pairs // 2))
    fig.patch.set_facecolor("#111111")
    fig.suptitle(f"Visual Audit  |  epsilon = {epsilon}",
                 color="white", fontsize=15, fontweight="bold", y=1.02)

    gs = gridspec.GridSpec(n_pairs // 2, 4, figure=fig,
                           hspace=0.6, wspace=0.3)

    for plot_i, idx in enumerate(selected):
        row, pair = divmod(plot_i, 2)
        col_clean = pair * 2
        col_adv   = pair * 2 + 1

        img_clean = X_test[idx].reshape(28, 28)
        img_adv   = X_adv[idx].reshape(28, 28)

        ax_c = fig.add_subplot(gs[row, col_clean])
        ax_c.imshow(img_clean, cmap="gray", vmin=0, vmax=1)
        ax_c.set_title(f"Clean  gt={y_test[idx]}  pred={orig_preds[idx]}",
                       color="#76ff03", fontsize=8)
        ax_c.axis("off")

        ax_a = fig.add_subplot(gs[row, col_adv])
        ax_a.imshow(img_adv, cmap="inferno", vmin=0, vmax=1)
        ax_a.set_title(f"Adversarial  pred={adv_preds[idx]}  (TRICKED)",
                       color="#ff1744", fontsize=8)
        ax_a.axis("off")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved -> {save_path}")


# ---------------------------------------------------------
# 6. MAIN PIPELINE
# ---------------------------------------------------------

def main():
    print("=" * 50)
    print("  PROBLEM 106 - THE ADVERSARIAL STRESS-TESTER")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_mnist(n_train=10000, n_test=2000)

    # Change to "rf" to use Random Forest instead
    USE_MODEL = "svm"

    if USE_MODEL == "svm":
        model = train_rbf_svm(X_train, y_train)
        model_name = "RBF-SVM"
    else:
        model = train_random_forest(X_train, y_train)
        model_name = "Random Forest"

    clean_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\n{model_name} Clean Accuracy: {clean_acc:.4f} ({clean_acc:.2%})")

    epsilons = np.linspace(0, 0.5, 26)
    eps_vals, accs, baseline, bp = compute_fragility_curve(
        model, X_test, y_test, epsilons=epsilons, attack="random"
    )
    plot_fragility_curve(eps_vals, accs, baseline, bp,
                         model_name=model_name,
                         save_path="fragility_curve.png")

    print("\nGenerating Visual Audit...")
    visual_audit(model, X_test, y_test,
                 epsilon=0.15, n_pairs=6,
                 save_path="visual_audit.png")

    print("\nDone. Check fragility_curve.png and visual_audit.png")


if __name__ == "__main__":
    main()
