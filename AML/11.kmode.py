#23BAI11010 Kmode
import os
import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_curve,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

CSV_PATH = r"R:\VS CODE\Dataset\Wholesale customers data.csv"
LABEL_COLUMN = "Channel"
OUTPUT_PREFIX = "kmodes_wholesale"
N_CLUSTERS = 3
BINS = 4  

def auto_discretize(df_numeric, bins=4):
    X = df_numeric.copy()
    for col in X.columns:
        try:
            X[col] = pd.qcut(X[col], q=bins, labels=False, duplicates="drop")
        except Exception:
            X[col] = pd.cut(X[col], bins=bins, labels=False)
    X = X.fillna(0).astype(int)
    return X

def align_labels(cluster_labels, true_labels, n_clusters):
    mapping = {}
    aligned = np.full_like(cluster_labels, fill_value=-1, dtype=int)
    true_arr = np.asarray(true_labels)
    for i in range(n_clusters):
        mask = (cluster_labels == i)
        if mask.sum() == 0:
            mapping[i] = -1
            continue
        vals = true_arr[mask]
        if vals.size == 0:
            mapping[i] = -1
            continue
        unique, counts = np.unique(vals, return_counts=True)
        mapped_label = unique[np.argmax(counts)]
        mapping[i] = int(mapped_label)
        aligned[mask] = int(mapped_label)
    return aligned, mapping

def plot_and_save_roc(y_true_bin, y_score, out_prefix, title="KModes ROC (one-vs-rest)"):
    plt.figure(figsize=(7, 6))
    aucs = {}
    n_classes = y_true_bin.shape[1]
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
        except Exception:
            fpr, tpr = [0, 1], [0, 1]
            auc = float("nan")
        aucs[i] = auc
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_roc.png")
    plt.close()
    return aucs

def main():
    df = pd.read_csv(CSV_PATH)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found. Columns: {df.columns.tolist()}")
    data = df.drop(columns=[LABEL_COLUMN])
    numeric = data.select_dtypes(include=[np.number])
    non_numeric = data.select_dtypes(exclude=[np.number])
    if numeric.shape[1] > 0:
        disc = auto_discretize(numeric, bins=BINS).astype(str)
    else:
        disc = pd.DataFrame()
    non_numeric_str = non_numeric.astype(str) if non_numeric.shape[1] > 0 else pd.DataFrame()
    X_cat = pd.concat([disc, non_numeric_str], axis=1)
    if X_cat.shape[1] == 0:
        raise ValueError("No features available for clustering after dropping label column.")
    km = KModes(n_clusters=N_CLUSTERS, init="Huang", n_init=5, verbose=0, random_state=42)
    preds = km.fit_predict(X_cat.values)
    y = df[LABEL_COLUMN].values
    aligned_preds, mapping = align_labels(preds, y, N_CLUSTERS)
    print("Cluster -> mapped label mapping:", mapping)
    acc = accuracy_score(y, aligned_preds)
    print("KModes accuracy:", acc)
    print(classification_report(y, aligned_preds, digits=4))
    cm = confusion_matrix(y, aligned_preds)
    print("Confusion matrix:\n", cm)
    classes = np.unique(y)
    n_classes = len(classes)
    y_bin = label_binarize(y, classes=classes)
    label_to_col = {label: idx for idx, label in enumerate(classes)}
    onehot = np.zeros((len(y), n_classes), dtype=int)
    for i, v in enumerate(aligned_preds):
        if v in label_to_col:
            onehot[i, label_to_col[v]] = 1
    if y_bin.shape[1] != n_classes:
        y_bin_full = np.zeros((len(y), n_classes), dtype=int)
        for i, val in enumerate(y):
            y_bin_full[i, label_to_col[val]] = 1
        y_bin_use = y_bin_full
    else:
        y_bin_use = y_bin
    aucs = plot_and_save_roc(y_bin_use, onehot, OUTPUT_PREFIX)
    print("AUCs:", aucs)
    summary = pd.DataFrame({"method": ["kmodes"], "accuracy": [acc]})
    summary.to_csv(OUTPUT_PREFIX + "_summary.csv", index=False)
    print(
        "Output files:",
        OUTPUT_PREFIX + "_confusion.csv",
        OUTPUT_PREFIX + "_roc.png",
        OUTPUT_PREFIX + "_summary.csv",
    )

if __name__ == "__main__":
    main()