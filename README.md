# 🛡️ Problem 106 — The Adversarial Stress-Tester

## Project Structure

```
adversarial_stress_tester.py   ← Main script (sklearn, no GPU needed)
bonus_fgsm_pytorch.py          ← Bonus: True FGSM with PyTorch CNN
README.md                      ← This file
```

---

## ⚙️ Setup

```bash
# Core requirements (main script)
pip install scikit-learn numpy matplotlib openml tqdm

# Bonus script (true FGSM)
pip install torch torchvision
```

---

## 🚀 Running

```bash
# Main pipeline (RBF-SVM, ~3-5 min on CPU)
python adversarial_stress_tester.py

# Bonus: PyTorch CNN with real gradient-based attack
python bonus_fgsm_pytorch.py
```

Switch between RBF-SVM and Random Forest by changing:
```python
USE_MODEL = "svm"   # or "rf"
```

---

## 📦 What Each Script Does

### `adversarial_stress_tester.py`
| Step | What happens |
|------|-------------|
| **1. Data** | Loads MNIST-784 from OpenML (10k train, 2k test) |
| **2. Model Hardening** | Trains RBF-SVM (C=10, γ=scale) inside a StandardScaler pipeline |
| **3. Perturbation Engine** | Adds ε × sign(random noise) to each pixel — FGSM-style without gradients |
| **4. Fragility Curve** | Sweeps ε ∈ [0, 0.5], plots Accuracy vs ε, marks Breaking Point |
| **5. Visual Audit** | Picks tricked samples, shows Clean vs Adversarial side-by-side |

### `bonus_fgsm_pytorch.py`
- Trains a 2-layer CNN (32→64 conv + 2 FC layers)
- Applies **true FGSM** using `loss.backward()` gradients
- Produces its own fragility curve and visual audit

---

## 💡 Ideas to Make This Project Better (Grade Boosters)

### 🔴 Advanced Attacks (High Impact)
1. **PGD (Projected Gradient Descent)** — Multi-step FGSM, far more powerful. The gold standard for adversarial evaluation.
2. **C&W Attack (Carlini & Wagner)** — Optimisation-based attack, finds minimal perturbations.
3. **Boundary Attack** — Decision-based (black-box), no gradients needed, works on sklearn models natively!
4. **Square Attack** — State-of-the-art black-box attack using random square-shaped updates.

### 🟡 Defenses (Distinguish your project)
5. **Adversarial Training** — Fine-tune model on generated adversarial examples. Check if it hardens the fragility curve.
6. **Feature Squeezing** — Apply image smoothing (median filter, bit-depth reduction) before inference to destroy perturbations.
7. **Gaussian Input Smoothing** — Certifiable defense via randomized smoothing (Cohen et al. 2019).
8. **Input Preprocessing Pipeline** — Compare: no defense vs. feature squeezing vs. adversarial training on the *same* fragility curve.

### 🟢 Analysis & Visualisation
9. **Perturbation Heatmap** — Show which pixels are most attacked (average |Δ| across samples). Great visual!
10. **Decision Boundary Visualization** — Use UMAP/t-SNE to show how adversarial samples cross boundaries.
11. **Per-class Breaking Point** — Does the model break faster on '8' vs '1'? Plot a fragility curve per digit class.
12. **Confidence vs. Epsilon** — If using SVM with `probability=True` or a CNN: show how model confidence collapses as ε grows.
13. **Transferability Test** — Attack trained on RF → test on SVM. Does the adversarial example transfer? (Spoiler: often yes!)

### 🔵 Engineering & Extras
14. **Interactive Epsilon Slider** — Use `matplotlib.widgets.Slider` or Gradio to let users drag ε and watch the image change live.
15. **Robustness Score Card** — Print a formatted table: model name, clean acc, breaking_eps, acc@ε=0.3, etc.
16. **Adversarial Dataset Export** — Save the adversarial examples as a new dataset (`.npy` files) for reuse.
17. **Streamlit Dashboard** — Wrap everything in a Streamlit app for a clickable demo.

---

## 📖 Key Concepts to Mention in Report

- **Adversarial robustness ≠ accuracy**: A 99% accurate model can still be completely fragile.
- **High-dimensional geometry**: In 784-dimensional pixel space, tiny moves in the "wrong direction" can jump across decision boundaries.
- **ε-ball**: All adversarial perturbations must stay within an ℓ∞ ball of radius ε around the original.
- **Black-box vs. White-box**: sklearn models = black-box (no gradients). PyTorch CNN = white-box (full FGSM).
- **Transferability**: Adversarial examples crafted for one model often fool a different model too.

---

## 📊 Expected Results

| Model | Clean Acc | Breaking Point (ε) |
|-------|-----------|-------------------|
| RBF-SVM | ~96–97% | ~0.15–0.25 |
| Random Forest | ~94–96% | ~0.20–0.30 |
| CNN (PyTorch, FGSM) | ~98–99% | ~0.10–0.20 |

*Exact values will vary. CNN with true FGSM breaks earlier because the attack is stronger.*
