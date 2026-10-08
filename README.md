# Adversarial Stress Test: Random Forest vs Logistic Regression

This project compares how two MNIST digit classifiers respond to increasing
random signed pixel noise. It measures noise robustness, not worst-case
adversarial robustness. Neither FGSM nor adversarial training is used.

## Original experiment

[`notebooks/MLjack.ipynb`](notebooks/MLjack.ipynb) is a byte-for-byte copy of the
user-supplied `MLjack (2).ipynb`, including its original code and saved outputs.
It is the source experiment for this repository. The unrelated SVM and CNN
scripts from the earlier repository version have been removed; Git history
preserves them.

The notebook reads `mnist_train.csv` and `mnist_test.csv`, expects a label in the
first column and 784 pixel columns, divides pixels by 255, and replaces missing
pixel values with zero. It trains a 100-tree Random Forest and Logistic
Regression with `solver='liblinear'` and `max_iter=100`.

Both models see the same noisy test images at each epsilon. Noise is generated
as `epsilon * sign(normal(0, 1))`, added to pixels, and clipped to `[0, 1]`.
The notebook does not seed that noise generator.

## Saved notebook results

These are historical outputs embedded in the supplied notebook, not newly
reproduced benchmarks. The original CSVs and environment versions were not
included, so exact reproduction has not been verified.

| Epsilon | Random Forest accuracy | Logistic Regression accuracy |
| ---: | ---: | ---: |
| 0.00 | 95.7% | 91.1% |
| 0.05 | 74.0% | 90.5% |
| 0.10 | 67.5% | 87.6% |
| 0.15 | 62.6% | 80.5% |
| 0.20 | 59.5% | 68.3% |
| 0.25 | 55.6% | 58.6% |
| 0.30 | 52.5% | 50.5% |
| 0.40 | 47.2% | 35.4% |
| 0.50 | 40.4% | 25.0% |

Separately printed clean accuracies are RF **95.71%** and LR **91.15%**.
RF leads on clean data; LR leads at the sampled noise levels from 0.05 through
0.25; RF leads again from 0.30 through 0.50. The later crossover is bracketed
between 0.25 and 0.30, not precisely measured at 0.28.

## Reading the notebook accurately

- "Hardening" in the original code means ordinary clean-data training.
- Random signed noise is not a targeted attack or a gradient-based attack.
- The plotted collapse line is half the RF baseline (47.855%), not 50%
  accuracy and not half of each model's own baseline.
- The visual audit shows the first test image with RF predictions only. It
  does not search for a successful misclassification or prove imperceptibility.
- Citation placeholders in the original notebook are unresolved references,
  not verified citations. They are retained only to preserve the source.
- Recent scikit-learn versions require an explicit one-vs-rest wrapper for
  multiclass Logistic Regression with `liblinear`; the runnable script will
  make that behavior explicit.
