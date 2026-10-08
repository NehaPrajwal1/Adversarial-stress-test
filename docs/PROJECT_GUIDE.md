# Project guide: Random Forest versus Logistic Regression

This guide follows the supplied MLjack notebook and Project Overview document.
The notebook code and its stored output establish what was implemented and
observed. Interpretations and new runner improvements are identified separately.

## 1. Explain the project in plain words

Imagine a computer reading a handwritten 7 from a form. It predicts correctly
on the original image. What happens if the image is corrupted? This experiment
gradually increases pixel noise and measures how two classifiers respond.

The question is: **does the model with higher clean accuracy also perform
better when inputs are noisy?** The historical results show that the answer
depends on the noise strength. There is no single winner across the tested range.

The project is an offline Python experiment, not a deployed service. It contains
no sensors, web backend, database, authentication system, SVM, or CNN in its
current implementation.

## 2. Input data and preprocessing

The original notebook reads two local CSV files with pandas. The first column
contains the digit label; the remaining 784 columns contain a flattened 28-by-28
grayscale image. The actual files were not supplied with the notebook, so their
row counts and exact dataset provenance are unverified. MNIST conventionally has
60,000 training and 10,000 test images, but those counts must not be attributed
to these particular CSVs without inspection.

An image is one row. A pixel is one feature. The label is the correct answer.

- X_train: training pixels, shape (number of training images, 784).
- y_train: training labels, shape (number of training images,).
- X_test and y_test: separate data used for evaluation.

Dividing each pixel by 255 changes its scale from 0-255 to 0-1:
0 becomes 0, 128 becomes approximately 0.502, and 255 becomes 1.
There is no StandardScaler in the original notebook or current runner.

SimpleImputer replaces missing pixel values with zero, corresponding to black.
It is fitted on training data and then applied to test data. This does not mean
missing values definitely occurred: it is a precaution in the source code.
Zero filling can alter an image and is not necessarily the best treatment for
every kind of missing data.

The runner validates column count, column order, labels and pixel ranges.
It explicitly preserves all 784 columns, including entirely missing ones.

## 3. The two models

### Random Forest

A decision tree asks a sequence of threshold questions, such as whether a pixel
is brighter than a learned value. A Random Forest combines many trees to reduce
dependence on a single tree. Randomness in training samples and feature selection
helps make the trees different.

The original uses RandomForestClassifier(n_estimators=100, random_state=42).
The 100 refers to trees, not images or classes. The seed controls the forest's
randomness. The original does not set n_jobs=-1.

A jury analogy is useful, but sklearn's classifier actually averages the trees'
class probabilities and chooses the largest; it is not strictly a majority vote
of each tree's winning label.

### Logistic Regression

Despite its name, Logistic Regression is a classifier. For a binary task, it
computes a weighted sum z = w*x + b and maps it through a sigmoid:

    probability = 1 / (1 + exp(-z))

Each weight expresses how a feature contributes to the score. The intercept
shifts the decision boundary. A linear boundary can be a line in two dimensions,
a plane in three, or a hyperplane in 784 dimensions.

There are ten digit classes. The original liblinear setup uses a one-vs-rest
approach: train a binary classifier for each digit against all other digits.
The final prediction comes from comparing the class scores/probabilities.
It is therefore misleading to describe the entire multiclass system as just
one line or one binary judge.

The original settings are max_iter=100 and solver='liblinear'. max_iter caps
optimizer iterations; it does not specify trees or epochs in a neural network.
A convergence warning means the optimizer may need more iterations or other
adjustments. The runner leaves warnings visible.

Current sklearn releases require explicit OneVsRestClassifier around liblinear
for multiclass data. The runner adds that wrapper while preserving the original
model family and one-vs-rest strategy. This does not guarantee identical numbers
across library versions.

### Why compare these models?

They represent contrasting decision structures: a tree ensemble and linear
class boundaries. Both are trained on the same clean training data and tested
on the same perturbed examples at each noise level.

That controls the input corruption for this comparison. It does not establish
that their hyperparameters or training budgets are optimal.

## 4. The noise mechanism, with an example

The original inject_noise function:

1. Draws independent standard-normal values with the same shape as the images.
2. Takes their signs, leaving approximately equally likely -1 and +1 values.
3. Multiplies those signs by epsilon.
4. Adds them to the original normalized pixels.
5. Clips the result to [0,1].

Example:

    Input:          [0.05, 0.40, 0.95]
    Random draws:   [-0.3, 1.7, 0.2]
    Signs:          [-1,   1,   1]
    Epsilon:        0.10
    Before clipping:[-0.05,0.50,1.05]
    After clipping: [0.00, 0.50,1.00]

Although the initial draws are Gaussian, the added noise is signed noise,
not continuous Gaussian noise. A draw of 0.01 and a draw of 100 produce the
same positive perturbation after sign(). An exactly zero draw produces zero.

The maximum absolute change to an individual pixel is at most epsilon:

    max(abs(noisy_image - clean_image)) <= epsilon

This is an L-infinity bound. It limits each pixel change, not total image change.
Clipping can make the actual change smaller than epsilon.

For a black pixel at 0, a negative change clips back to 0 while a positive
change becomes epsilon. Therefore, after clipping, the corruption is not
zero-mean near the boundaries. Many MNIST background pixels are black, so this
effect matters when interpreting the experiment.

### What this method is not

It is not FGSM. It does not compute gradients, optimize loss, choose a target
digit, or search using predictions. It is model-agnostic random corruption.

FGSM uses the sign of an input-loss gradient to take one locally motivated
attack step. It does not guarantee the globally worst possible modification.
Random Forest predictions are piecewise constant with discontinuities at splits,
so ordinary input gradients are not a useful attack direction. Logistic
Regression has differentiable scores and loss, but this project does not use
those gradients.

The original terms "targeted nudge" and "hardening" overstate the implementation.
Both models are trained on clean data; no adversarial defense is trained.

## 5. Evaluation and saved results

The epsilon grid is:

    0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50

At each level the original generates one noisy test array, then passes that
same array to RF and LR. A new random draw is used at each epsilon. Each level
starts from the clean test data; perturbations are not cumulatively added.

Accuracy is the fraction of predictions equal to the correct labels.
The fragility curve plots this accuracy against epsilon.

The saved output records clean accuracies of RF 95.71% and LR 91.15%.
Full rounded sweep values are in ../results/notebook_results.csv.

- RF wins on clean data.
- At epsilon 0.05, RF scores 74.0% and LR scores 90.5%.
- LR remains ahead at every sampled level through 0.25.
- At 0.30, RF scores 52.5% and LR scores 50.5%.
- At 0.50, RF scores 40.4% and LR scores 25.0%.

The later ordering reversal is bracketed by epsilon 0.25 and 0.30.
Linear interpolation of the rounded numbers suggests approximately 0.28,
but that epsilon was not evaluated. There is also an initial ordering reversal
between clean data and 0.05.

Do not say LR "barely drops throughout low noise": by 0.25 it has fallen to
58.6%. Do not claim RF reaches a plateau; it continues declining across the
recorded high-noise points.

These outputs belong to one historical notebook run. Its noise was unseeded,
and the source CSVs and full environment are unavailable. They are useful
observations, not independently reproduced benchmarks or statistical guarantees.

### Why might the patterns differ?

Tree threshold decisions can change abruptly as pixels move across splits.
Linear class scores respond through weighted sums; some independent signed
changes may cancel. These are plausible explanations, not verified causes.

Clipping introduces bias near 0 and 1, pixel weights differ, and trees can have
correlated errors. There is no evidence that "first tree questions" specifically
cause the early drop, or that ensemble voting prevents failure at high epsilon.

To investigate mechanisms, compare prediction margins, tree disagreement,
feature distributions after clipping, per-class failures and repeated seeds.

## 6. The collapse threshold

The original draws one line at RF clean accuracy divided by two:

    0.9571 / 2 = 0.47855, approximately 47.855%

This is neither literal 50% accuracy nor a separately normalized threshold
for LR. The notebook draws the line but does not calculate crossing values.

The runner uses one threshold per model. For the historical LR baseline:

    0.9115 / 2 = 0.45575, approximately 45.575%

Using the printed historical values, both models first fall below their own
half-baseline threshold at the sampled epsilon 0.40. Their values at 0.30
remain above their respective thresholds.

This is a project-defined summary, not a universal robustness metric.
A crossing is the first tested point satisfying the condition, not an exact
critical epsilon. No crossing is represented by null in the runner report.

## 7. The original visual audit

The original selects test sample zero and draws separate noise at epsilon 0.2.
It shows only RF predictions. The stored image has prediction 7 on both sides.

Therefore the stored audit is a corruption example, not a successful attack
demonstration. The true label is not printed in that original figure.
The corruption is visible; it is not evidence of imperceptibility.

The runner keeps sample zero as an explicit inspection example, uses matching
grayscale limits, and shows the true label plus both models' predictions.
It does not label the sample as fooled unless the predictions establish that.
The audit uses a separate seeded draw and is not the exact sweep image.

## 8. Original experiment versus the new runner

| Aspect | Original notebook | Runnable version |
|---|---|---|
| Models | RF 100 trees; LR liblinear, 100 iterations | Same defaults |
| Multiclass LR | Implicit historical OvR behavior | Explicit OvR wrapper |
| Normalization | Divide by 255 | Same |
| Missing pixels | Constant zero imputation | Same; preserve empty columns |
| Noise | Sign of normal draws | Same distribution |
| Noise reproducibility | Global NumPy RNG, unseeded | Local Generator, seed 42 default |
| Within-level comparison | Same noise for both models | Same |
| Threshold | One RF-based line | Each model's own threshold |
| Audit | First image, RF predictions | First image, both predictions and label |
| Outputs | Notebook prints and plots | CSV, JSON, PNG |
| Validation | Implicit assumptions | Explicit CSV and range checks |
| Automated tests | None | Offline unittest suite |

Setting a seed now cannot reconstruct the original unseeded noise. Re-running
with another seed or library version can produce different results.

The JSON report records dataset hashes, row counts, model settings and library
versions. It does not contain the raw CSV data. Reusing an output directory
overwrites previous artifacts, so use a new directory for each experiment.

## 9. Terms to know for an interview

| Term | Meaning in this project |
|---|---|
| Feature | One pixel intensity |
| Label / ground truth | Correct digit |
| Training | Learn model parameters from labeled images |
| Inference | Predict using a trained model |
| Baseline | Accuracy before corruption |
| Ensemble | Combination of multiple models, here trees |
| Decision boundary | Where the predicted class changes |
| Hyperparameter | Setting such as tree count or iteration limit |
| Perturbation | Input modification |
| Epsilon | Maximum intended per-pixel noise magnitude |
| Clipping | Keep values in a valid range |
| Seed | Starting state for a pseudorandom generator |
| Overfitting | Learning details that do not generalize |
| Data leakage | Evaluation information influencing development improperly |
| One-vs-rest | One binary classifier per class against all other classes |
| Gradient | Local sensitivity of a function to its inputs |
| Threat model | What an attacker can know and change |
| Evasion | Manipulate prediction-time inputs |
| Poisoning | Manipulate training data |
| Corruption robustness | Performance under specified accidental/random disturbances |
| Adversarial robustness | Performance against deliberate failure-seeking perturbations |
| Reproducibility | Repeat an experiment with known data, settings and environment |

## 10. Interview answers grounded in this implementation

**Explain the project in 90 seconds.**

"I compared a Random Forest and Logistic Regression on handwritten digit images.
After normalizing pixels and filling missing values, I trained both on clean
data. I then added random signed pixel noise at increasing strengths, gave both
models the same noisy examples, and plotted their accuracy curves. In the
notebook's stored run, RF had higher clean accuracy, LR performed better from
epsilon 0.05 through 0.25, and RF performed better again from 0.30 onwards.
The takeaway is that model rankings depend on the evaluation conditions.
This was random-noise testing, not FGSM or a proof of adversarial robustness."

Use first-person contribution claims only for work you actually performed.

**Why did you choose these models?**

"They contrast a nonlinear tree ensemble with linear multiclass boundaries.
Both support the same tabular pixel representation, allowing a shared input
corruption experiment."

**Why does higher clean accuracy not establish robustness?**

"Clean accuracy samples one input condition. Corruption can affect different
decision structures differently. The observed ranking changes with epsilon."

**Is this a fair comparison?**

"Both models receive the same training data, preprocessing and noisy examples
within each epsilon. But the experiment is one run, has no uncertainty estimates
or systematic hyperparameter tuning, and evaluates only one corruption family."

**Does random_state=42 make the original notebook fully reproducible?**

"No. It seeds the forest, but the notebook's NumPy noise is unseeded. The new
runner explicitly seeds the noise generator and records versions and hashes."

**Did your visual audit fool the model?**

"The stored audit has RF prediction 7 before and after corruption. It is a
visual inspection example, not evidence of a successful attack."

**What does a 50% collapse mean?**

"Half of a model's starting accuracy. The original draws only the RF threshold;
the runner now calculates each model's own first sampled crossing."

**What would you improve scientifically?**

"Repeat seeds, report variability, use finer epsilon steps near reversals,
analyze per-class errors, and evaluate additional corruption and attack methods.
Use held-out validation data for tuning and leave the test set for evaluation."

**What would you improve as an engineer?**

"Persist model and preprocessing artifacts, add experiment IDs and configuration
files, measure runtime and memory, and package a job-based API if deployment is
needed. Those are future extensions, not existing functionality."

**How would you test it?**

"Check epsilon-zero identity, pixel bounds, the L-infinity budget, fixed-seed
repeatability, shared inputs, missing-data behavior, and threshold calculations.
Run a small end-to-end fixture to verify saved artifacts and multiclass support."

**How could this apply to sensors?**

"The reusable idea is controlled input disturbance. Sensor tests would use
physical units, bias, drift, spikes, missing readings and timing errors.
Pixel epsilon values cannot be copied directly to telemetry."

**What is the main limitation?**

"The historical outputs are one unseeded run without the source CSVs or full
environment, and random corruption does not find worst-case adversarial inputs."

## 11. Validation performed on the restored repository

The offline suite passes 16 tests, including a three-class synthetic CSV
end-to-end run that trains both models and writes all output artifacts.
The original notebook was checked for byte-for-byte preservation and Python
source syntax. The stored plots were visually inspected.

This validates implementation mechanics, not the historical MNIST accuracies.
The notebook was not re-executed on MNIST because the original CSVs were absent.
