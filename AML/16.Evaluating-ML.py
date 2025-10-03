import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.datasets import fetch_openml 
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt

# --- 1. Load and Prepare Imbalanced Data ---
def load_and_prepare_data(test_size=0.3, random_state=42):
    print("Loading Blood Transfusion dataset (Imbalanced) and preparing for evaluation...")
    
    try:
        data = fetch_openml(data_id=1464, as_frame=True, parser='auto')
    except Exception as e:
        print(f"Error loading dataset: {e}. Falling back to dummy data.")
        X = np.random.rand(700, 4)
        y_encoded = np.concatenate([np.zeros(600, dtype=int), np.ones(100, dtype=int)])
    else:
        X = data.data.to_numpy()
        y = data.target.astype(int).to_numpy() 
        unique_classes = np.unique(y)
        if len(unique_classes) == 2 and (unique_classes == [1, 2]).all():

            y_encoded = np.where(y == 2, 1, 0)
        else:
            y_encoded = y

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    class_counts = np.bincount(y_encoded)

    if class_counts[0] > 0:
        minority_ratio = class_counts[1] / class_counts[0] * 100
    else:
        minority_ratio = np.inf
    
    print(f"Total samples used: {len(X)}")
    print(f"Features (Dimension): {X.shape[1]}")
    print(f"Classes (0/1): {class_counts[0]} / {class_counts[1]}")
    print(f"Imbalance Ratio (Class 1 to Class 0): {minority_ratio:.2f}%")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print("-" * 50)
    target_names = ['Did Not Donate (Class 0)', 'Did Donate (Class 1)']
    
    return X_train, X_test, y_train, y_test, target_names

# --- 2. Training Scenarios ---

def train_naive_model(X_train, y_train): 
    print(f"Training Naive Logistic Regression (No Imbalance Handling)...")
    model = LogisticRegression(random_state=42, max_iter=200) 
    model.fit(X_train, y_train)
    return model

def train_balanced_model(X_train, y_train): 
    """Trains Logistic Regression with class_weight='balanced' (Imbalance Handling)."""
    print(f"Training Balanced Logistic Regression (Class Weights Applied)...")
    model = LogisticRegression(random_state=42, max_iter=200, class_weight='balanced') 
    model.fit(X_train, y_train)
    return model

# --- 3. Evaluate using Imbalance-Robust Metrics ---
def evaluate_model(model, X_test, y_test, target_names, model_name):
    # Prediction of classes
    y_pred = model.predict(X_test)
    
    # Prediction of probabilities (required for ROC AUC)
    y_proba = model.predict_proba(X_test)[:, 1] 
    
    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print("-" * 50)
    print(f"Model Evaluation: {model_name}")
    print(f"1. Accuracy Score (Overall Correctness): {accuracy:.4f}") 
    print(f"2. ROC AUC Score (Separation Capability): {roc_auc:.4f}") 
    print("\n3. Classification Report (Detailed Imbalance Metrics):")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    # 4. Plot ROC Curve (Visualizes performance across all thresholds)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f'Receiver Operating Characteristic ({model_name})')
    plt.legend(loc="lower right")
    plt.show()
    print(" (Check the plot window for ROC curve visualization.)")
    print("-" * 50)
    
# --- Main Execution ---
if __name__ == '__main__':
    # Step 1: Load and Split Data
    X_train, X_test, y_train, y_test, target_names = load_and_prepare_data()
    
    # SCENARIO 1: Naive Training (No Imbalance Handling)
    naive_model = train_naive_model(X_train, y_train) 
    evaluate_model(naive_model, X_test, y_test, target_names, "Naive Model")
    
    # SCENARIO 2: Balanced Training (Imbalance Handling)
    balanced_model = train_balanced_model(X_train, y_train) 
    evaluate_model(balanced_model, X_test, y_test, target_names, "Balanced Model (Class Weights)")
    
    # Discussion points (printed to console):
    print("\n--- Evaluation Insight ---")
    print("In the Naive Model, notice that Accuracy might be high (because the model easily predicts the majority class correctly), but the 'Recall' for Class 1 (Did Donate) is likely very low. This means the model ignores the minority class.")
    print("In the Balanced Model, the 'Recall' for Class 1 should be significantly higher, meaning the model is better at finding the donors, though the overall accuracy might drop slightly due to more false positives in the majority class.")