#23BAI11010 PCA
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DATA_PATH = r"R:\VS CODE\Dataset\Wholesale customers data.csv"
OUTPUT_TRANSFORMED = "transformed_data.csv"
N_COMPONENTS = None  
TARGET_COLUMN = None  
RANDOM_STATE = 42

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    return df

def prepare_features(df, target_col=None):
    if target_col is not None and target_col in df.columns:
        y = df[target_col].copy()
        X = df.drop(columns=[target_col]).copy()
    else:
        y = None
        X = df.copy()
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        raise ValueError("No numeric columns found for PCA.")
    return X, y

def impute_and_scale(X):
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled, imputer, scaler

def run_pca(X_scaled, n_components=None, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca

def print_pca_summary(pca):
    ev = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    cum_var = np.cumsum(evr)

    print("PCA Summary:")
    print(f" Number of components: {pca.n_components_}")
    print(" Explained variance (per component):")
    for i, (v, r, c) in enumerate(zip(ev, evr, cum_var), start=1):
        print(f"  PC{i}: eigenvalue={v:.4f}, variance_ratio={r:.4f}, cumulative={c:.4f}")

def plot_scree(pca, save_path="scree_plot.png"):
    evr = pca.explained_variance_ratio_
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(evr)+1), evr, marker='o')
    plt.title("Scree Plot — Explained Variance Ratio")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.xticks(np.arange(1, len(evr)+1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)   # <<< saves image
    plt.show()
    print(f"Scree plot saved as {save_path}")

def plot_first_two(X_pca, y=None, save_path="pca_scatter.png"):
    if X_pca.shape[1] < 2:
        print("Less than two principal components — skipping 2D scatter.")
        return
    plt.figure(figsize=(6,6))
    if y is None:
        plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.7)
    else:
        unique = np.unique(y)
        if len(unique) <= 10:
            for val in unique:
                mask = (y == val)
                plt.scatter(X_pca[mask,0], X_pca[mask,1], label=str(val), alpha=0.7)
            plt.legend(title="target")
        else:
            plt.scatter(X_pca[:,0], X_pca[:,1], c=y, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA: First two principal components")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)   # <<< saves image
    plt.show()
    print(f"PCA scatter saved as {save_path}")

def save_transformed(X_pca, feature_names=None, y=None, outpath=OUTPUT_TRANSFORMED):
    cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    df_out = pd.DataFrame(X_pca, columns=cols)
    if y is not None:
        df_out["target"] = y.reset_index(drop=True)
    df_out.to_csv(outpath, index=False)
    print(f"Transformed data saved to: {outpath}")

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Data shape: {df.shape}")
    X, y = prepare_features(df, target_col=TARGET_COLUMN)
    print("Imputing and scaling...")
    X_scaled, imputer, scaler = impute_and_scale(X)
    print("Running PCA...")
    pca, X_pca = run_pca(X_scaled, n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    print_pca_summary(pca)
    plot_scree(pca)
    plot_first_two(X_pca, y=y)
    save_transformed(X_pca, feature_names=X.columns.tolist(), y=y, outpath=OUTPUT_TRANSFORMED)

if __name__ == "__main__":
    main()