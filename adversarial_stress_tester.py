"""Random Forest / Logistic Regression noise stress test, based on MLjack.ipynb.

This is random signed-noise evaluation, not FGSM or adversarial training.
The original notebook and saved outputs are preserved in notebooks/MLjack.ipynb.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a desktop display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

EPSILONS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


def read_mnist_csv(path):
    """Read a headered CSV: label first, then exactly 784 pixels in [0,255]."""
    frame = pd.read_csv(path)
    if frame.empty or frame.shape[1] != 785:
        raise ValueError(f"{path}: expected nonempty rows with 1 label + 784 pixels.")
    # Reject common headerless MNIST files rather than silently dropping row 1.
    try:
        float(str(frame.columns[0]))
    except ValueError:
        pass
    else:
        raise ValueError(f"{path}: a header row is required; first column is the label.")
    try:
        values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path}: labels and pixels must be numeric.") from exc
    labels, pixels = values[:, 0], values[:, 1:]
    if not np.all(np.isfinite(labels) & (labels == np.floor(labels))
                  & (labels >= 0) & (labels <= 9)):
        raise ValueError(f"{path}: labels must be nonmissing integers from 0 to 9.")
    if np.isinf(pixels).any() or np.any((pixels < 0) | (pixels > 255)):
        raise ValueError(f"{path}: pixels must be in [0,255] or missing.")
    return pixels / 255.0, labels.astype(int), tuple(frame.columns[1:])


def load_data(train_path, test_path):
    """Normalize then fit zero imputation on training data only."""
    X_train, y_train, columns = read_mnist_csv(train_path)
    X_test, y_test, test_columns = read_mnist_csv(test_path)
    if columns != test_columns:
        raise ValueError("Training and test pixel columns must have identical order.")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training data must contain at least two digit classes.")
    if not set(y_test).issubset(set(y_train)):
        raise ValueError("Test labels include classes absent from training data.")
    # Keep all 784 features even when a training column is entirely missing.
    imputer = SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)
    return (imputer.fit_transform(X_train), imputer.transform(X_test),
            y_train, y_test)


def train_models(X_train, y_train, seed=42, trees=100, max_iter=100):
    """Preserve the notebook's forest size and liblinear one-vs-rest approach."""
    models = {
        "random_forest": RandomForestClassifier(n_estimators=trees, random_state=seed),
        # Explicit OvR supports modern sklearn, where liblinear is binary-only.
        "logistic_regression": OneVsRestClassifier(
            LogisticRegression(solver="liblinear", max_iter=max_iter, random_state=seed)
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def inject_noise(X, epsilon, rng):
    """Add epsilon * sign(N(0,1)) and clip; do not modify the input."""
    X = np.asarray(X)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative.")
    if not np.isfinite(X).all() or np.any((X < 0) | (X > 1)):
        raise ValueError("Noise input must be finite and normalized to [0,1].")
    noise = rng.normal(0, 1, X.shape)
    return np.clip(X + epsilon * np.sign(noise), 0, 1)


def validate_epsilons(epsilons):
    values = np.asarray(epsilons, dtype=float)
    if (values.ndim != 1 or values.size == 0 or not np.isfinite(values).all()
            or values[0] != 0 or np.any(np.diff(values) <= 0)):
        raise ValueError("epsilons must start at 0 and increase strictly.")
    return values


def first_half_baseline_crossing(epsilons, accuracies, baseline):
    """Return first sampled epsilon at/below half baseline, or None."""
    for epsilon, accuracy in zip(epsilons, accuracies):
        if accuracy <= baseline / 2:
            return float(epsilon)
    return None


def evaluate(models, X_test, y_test, epsilons=EPSILONS, seed=42):
    """Use the exact same perturbed test array for both models at each epsilon."""
    epsilons = validate_epsilons(epsilons)
    if len(y_test) == 0:
        raise ValueError("Test data must not be empty.")
    clean_predictions = {name: model.predict(X_test) for name, model in models.items()}
    baseline = {name: float(accuracy_score(y_test, pred))
                for name, pred in clean_predictions.items()}
    rng = np.random.default_rng(seed)
    rows = []
    for epsilon in epsilons:
        X_noisy = inject_noise(X_test, epsilon, rng)
        for name, model in models.items():
            prediction = model.predict(X_noisy)
            initially_correct = clean_predictions[name] == y_test
            count = int(initially_correct.sum())
            failures = int(np.sum(initially_correct & (prediction != y_test)))
            rows.append({
                "epsilon": float(epsilon),
                "model": name,
                "accuracy": float(accuracy_score(y_test, prediction)),
                "clean_accuracy": baseline[name],
                "initially_correct": count,
                "correct_to_wrong": failures,
                # This is corruption-induced failure rate, not an optimized attack ASR.
                "failure_rate_on_clean_correct": failures / count if count else None,
            })
    return pd.DataFrame(rows), baseline


def plot_curve(metrics, baseline, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, color in (("random_forest", "tab:blue"),
                        ("logistic_regression", "tab:orange")):
        rows = metrics[metrics.model == name]
        label = name.replace("_", " ").title()
        ax.plot(rows.epsilon, rows.accuracy, "o-", color=color, label=label)
        ax.axhline(baseline[name] / 2, color=color, linestyle="--",
                   label=f"{label}: half clean accuracy")
    ax.set(xlabel="Epsilon (random signed-noise strength)", ylabel="Accuracy",
           title="MNIST noise stress test: Random Forest vs Logistic Regression",
           ylim=(0, 1))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_audit(models, X_test, y_test, path, epsilon=0.2, sample_index=0, seed=42):
    """Inspect the first sample like the notebook; do not promise it is fooled."""
    clean = X_test[sample_index:sample_index + 1]
    noisy = inject_noise(clean, epsilon, np.random.default_rng(seed))
    predictions = {
        name: {"clean": int(model.predict(clean)[0]),
               "noisy": int(model.predict(noisy)[0])}
        for name, model in models.items()
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, pixels, key, title in (
        (axes[0], clean, "clean", "Clean"),
        (axes[1], noisy, "noisy", f"Random signed noise, epsilon={epsilon}"),
    ):
        details = "\n".join(
            f"{name.replace('_', ' ').title()}: {pred[key]}"
            for name, pred in predictions.items()
        )
        ax.imshow(pixels.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title}\nTrue label: {y_test[sample_index]}\n{details}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"sample_index": sample_index, "true_label": int(y_test[sample_index]),
            "epsilon": epsilon, "seed": seed,
            "note": "Separate noise draw; not a selected failure from the sweep.",
            "predictions": predictions}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("mnist_train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("mnist_test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args(argv)
    if args.seed < 0 or args.trees < 1 or args.max_iter < 1:
        parser.error("seed must be nonnegative; trees and max-iter must be positive.")
    try:
        X_train, X_test, y_train, y_test = load_data(args.train_csv, args.test_csv)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.error(str(exc))

    print(f"Training RF and LR on {len(y_train)} rows; testing {len(y_test)} rows.")
    models = train_models(X_train, y_train, args.seed, args.trees, args.max_iter)
    metrics, baseline = evaluate(models, X_test, y_test, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    plot_curve(metrics, baseline, args.output_dir / "fragility_curve.png")
    audit = plot_audit(models, X_test, y_test, args.output_dir / "visual_audit.png",
                       seed=args.seed)
    breaking = {}
    for name in models:
        rows = metrics[metrics.model == name]
        breaking[name] = first_half_baseline_crossing(
            rows.epsilon, rows.accuracy, baseline[name])
    report = {
        "method": "random signed noise; not FGSM",
        "seed": args.seed,
        "training_rows": len(y_train),
        "test_rows": len(y_test),
        "features": X_train.shape[1],
        "epsilons": list(EPSILONS),
        "models": {"random_forest": {"n_estimators": args.trees},
                   "logistic_regression": {"strategy": "one-vs-rest",
                                           "solver": "liblinear",
                                           "max_iter": args.max_iter}},
        "clean_accuracy": baseline,
        "half_baseline_threshold": {name: value / 2 for name, value in baseline.items()},
        "first_sampled_half_baseline_crossing": breaking,
        "datasets": {"train_sha256": file_sha256(args.train_csv),
                     "test_sha256": file_sha256(args.test_csv)},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
        "visual_audit": audit,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"Saved metrics, report and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
