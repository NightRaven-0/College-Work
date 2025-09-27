#23BAI11010 Kmeans
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_curve,
    roc_auc_score,
)
import matplotlib.pyplot as plt

CSV_PATH = r"R:\VS CODE\Dataset\Wholesale customers data.csv" 
LABEL_COLUMN = "Channel"
OUTPUT_PREFIX = "kmeans_wholesale"
N_CLUSTERS = 3

def align_labels(cluster_labels, true_labels, n_clusters):
    mapping = {}
    aligned = np.full_like(cluster_labels, fill_value=-1, dtype=int)
    for i in range(n_clusters):
        mask = (cluster_labels == i)
        if mask.sum() == 0:
            mapping[i] = -1
            continue
        vals = np.asarray(true_labels)[mask]
        if vals.size == 0:
            mapping[i] = -1
            continue
        unique, counts = np.unique(vals, return_counts=True)
        mapped_label = unique[np.argmax(counts)]
        mapping[i] = int(mapped_label)
        aligned[mask] = int(mapped_label)
    return aligned, mapping

def plot_and_save_roc(y_true_bin, y_score, out_prefix):
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
    plt.title("KMeans ROC (one-vs-rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_roc.png")
    plt.close()
    return aucs

def main():
    df = pd.read_csv(CSV_PATH)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Label column '{LABEL_COLUMN}' not found. Columns: {df.columns.tolist()}"
        )
    numeric = df.select_dtypes(include=[np.number]).copy()
    if LABEL_COLUMN in numeric.columns:
        numeric = numeric.drop(columns=[LABEL_COLUMN])
    X = numeric.dropna()
    y = df.loc[X.index, LABEL_COLUMN].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    preds = kmeans.fit_predict(Xs)
    aligned_preds, mapping = align_labels(preds, y, N_CLUSTERS)
    print("Cluster -> mapped label mapping:", mapping)
    acc = accuracy_score(y, aligned_preds)
    print("KMeans accuracy:", acc)
    print(classification_report(y, aligned_preds, digits=4))
    cm = confusion_matrix(y, aligned_preds)
    print("Confusion matrix:\n", cm)
    classes = np.unique(y)
    y_bin = label_binarize(y, classes=classes) 
    n_classes = len(classes)
    onehot = np.zeros((len(y), n_classes), dtype=int)
    label_to_col = {label: idx for idx, label in enumerate(classes)}

    for i, v in enumerate(aligned_preds):
        if v in label_to_col:
            onehot[i, label_to_col[v]] = 1
        else:
            pass
    aucs = plot_and_save_roc(y_bin if y_bin.shape[1] == n_classes else
                             np.eye(n_classes)[[label_to_col[val] for val in y]],
                             onehot, OUTPUT_PREFIX)
    print("AUCs:", aucs)
    summary = pd.DataFrame({"method": ["kmeans"], "accuracy": [acc]})
    summary.to_csv(OUTPUT_PREFIX + "_summary.csv", index=False)
    print("Output files:",
          OUTPUT_PREFIX + "_confusion.csv",
          OUTPUT_PREFIX + "_roc.png",
          OUTPUT_PREFIX + "_summary.csv")
if __name__ == "__main__":
    main()