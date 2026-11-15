"""
Statistical analysis for the flow-anomaly detector.

The project's whole premise is: train a model that flags when flow has
stopped (clog) or is leaking, and back that flag with statistics strong
enough to trust a threshold on it. This module supplies exactly that layer
on top of classify.py's models:

  - ROC / precision-recall curves and AUC for the binary "is something
    wrong with this vessel tree at all" detector
  - sensitivity/specificity at a chosen operating threshold, plus the
    Youden's-J-optimal threshold
  - stratified k-fold cross-validation with a 95% CI on accuracy, so a
    single lucky train/test split can't be mistaken for real performance
  - a bootstrap CI on held-out accuracy
  - a permutation test: is the trained model's accuracy actually higher
    than a model trained on label noise would get, or could this be chance?
  - McNemar's test, a paired significance test for "is classifier A really
    better than classifier B on the same test points, or is the gap noise?"
"""

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


# ---------------------------------------------------------------- ROC / PR --

def roc_pr_analysis(y_true, y_proba):
    """ROC and precision-recall curves + AUCs for a binary 0/1 target."""
    fpr, tpr, roc_thresh = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    precision, recall, pr_thresh = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    return {
        "fpr": fpr, "tpr": tpr, "roc_thresholds": roc_thresh, "roc_auc": roc_auc,
        "precision": precision, "recall": recall, "pr_thresholds": pr_thresh,
        "pr_auc": pr_auc,
    }


def threshold_sweep(y_true, y_proba, thresholds=None):
    """Sensitivity/specificity at a grid of thresholds, plus the
    Youden's-J-optimal operating point (max of sensitivity + specificity - 1).
    This is literally the "thresholded and upheld by statistics" step: it
    picks the probability cutoff above which the detector calls "anomaly".
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    y_true = np.asarray(y_true)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0   # recall / true positive rate
        spec = tn / (tn + fp) if (tn + fp) else 0.0   # true negative rate
        rows.append((t, sens, spec, sens + spec - 1))

    rows = np.array(rows)
    best_idx = int(np.argmax(rows[:, 3]))
    return {
        "thresholds": rows[:, 0],
        "sensitivity": rows[:, 1],
        "specificity": rows[:, 2],
        "youden_j": rows[:, 3],
        "best_threshold": float(rows[best_idx, 0]),
        "best_sensitivity": float(rows[best_idx, 1]),
        "best_specificity": float(rows[best_idx, 2]),
    }


# ------------------------------------------------------------ uncertainty --

def bootstrap_accuracy_ci(y_true, y_pred, n_boot=2000, ci=0.95, seed=0):
    """Nonparametric bootstrap CI on held-out accuracy: resample the test
    set with replacement n_boot times and take the central `ci` interval of
    the resulting accuracy distribution."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    accs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[b] = accuracy_score(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(accs, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"mean": float(accs.mean()), "lo": float(lo), "hi": float(hi), "samples": accs}


def cross_validated_accuracy(model, X, y, k=5, seed=0):
    """Stratified k-fold CV accuracy with a normal-approximation 95% CI on
    the mean, so performance isn't reported from one arbitrary split."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        m = clone(model)
        m.fit(X[train_idx], np.asarray(y)[train_idx])
        pred = m.predict(X[test_idx])
        scores.append(accuracy_score(np.asarray(y)[test_idx], pred))
    scores = np.array(scores)
    se = scores.std(ddof=1) / np.sqrt(k)
    me = 1.96 * se
    return {
        "scores": scores, "mean": float(scores.mean()),
        "lo": float(scores.mean() - me), "hi": float(scores.mean() + me),
    }


def permutation_test_accuracy(model, X_train, y_train, X_test, y_test,
                              n_perm=200, seed=0):
    """Is the trained model's test accuracy better than chance? Retrain
    n_perm times on label-shuffled training data, build a null distribution
    of test accuracy, and report the true accuracy's empirical p-value."""
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    m = clone(model)
    m.fit(X_train, y_train)
    true_acc = accuracy_score(y_test, m.predict(X_test))

    null_accs = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(y_train)
        m_null = clone(model)
        m_null.fit(X_train, shuffled)
        null_accs[i] = accuracy_score(y_test, m_null.predict(X_test))

    p_value = (np.sum(null_accs >= true_acc) + 1) / (n_perm + 1)
    return {
        "true_accuracy": float(true_acc),
        "null_mean": float(null_accs.mean()),
        "null_std": float(null_accs.std()),
        "null_accs": null_accs,
        "p_value": float(p_value),
    }


def mcnemar_test(y_true, pred_a, pred_b):
    """Paired significance test for whether two classifiers' error rates
    differ on the *same* test points (e.g. Random Forest vs SVM)."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    # n01: A wrong, B right.  n10: A right, B wrong.
    n01 = int(np.sum((~a_correct) & b_correct))
    n10 = int(np.sum(a_correct & (~b_correct)))

    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "statistic": 0.0, "p_value": 1.0}

    # exact binomial form (robust for small discordant counts)
    p_value = 2 * scipy_stats.binom.cdf(min(n01, n10), n, 0.5)
    p_value = min(p_value, 1.0)
    statistic = (abs(n01 - n10) - 1) ** 2 / n  # continuity-corrected chi-square stat
    return {"n01": n01, "n10": n10, "statistic": float(statistic), "p_value": float(p_value)}
