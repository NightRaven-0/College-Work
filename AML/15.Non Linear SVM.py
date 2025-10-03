import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.datasets import fetch_openml #
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler 
from sklearn.impute import SimpleImputer

# --- 1. Load and Prepare Real-World Data ---
def load_and_prepare_data(test_size=0.3, random_state=42):
    print("Loading Optical Recognition of Handwritten Digits dataset and preparing for Non-Linear SVM...")
    

    try:
        data = fetch_openml(name='optdigits', version=1, as_frame=True, parser='auto')
    except Exception as e:
        print(f"Error loading Optdigits dataset: {e}. Please check your network connection.")

        X = np.random.rand(500, 64)
        y = np.random.randint(0, 10, 500)
        target_names = [str(i) for i in range(10)]
        X_imputed = X
        y_encoded = y
    else:
        X = data.data.to_numpy()
        y = data.target.astype(int).to_numpy()
        
        X_imputed = X
        y_encoded = y
        target_names = [str(c) for c in np.unique(y_encoded)] 

    # Split the data

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_scaled_full = scaler.transform(X_imputed) 

    print(f"Total samples used: {len(X_imputed)}")
    print(f"Features (Dimension): {X_imputed.shape[1]} (8x8 pixel values)")
    print(f"Target Classes: {len(target_names)} (Digits {target_names[0]} to {target_names[-1]})")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("-" * 50)
    
    return X_train, X_test, y_train, y_test, target_names, X_scaled_full 

# --- 2. Initialize and Train the Non-Linear SVM Classifier ---
def train_nonlinear_svm(X_train, y_train): 
    print(f"Training Non-Linear SVM Classifier (RBF Kernel)...")
    
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42) 
    
    model.fit(X_train, y_train)
    print("Training complete.")
    return model

# --- 3. Predict, Evaluate, and Demonstrate ---
def evaluate_and_predict(model, X_test, y_test, target_names, X_full_scaled):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print("-" * 50)
    print("Model Evaluation (Non-Linear SVM / RBF Kernel):")
    print(f"Accuracy Score on Test Data: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    print("-" * 50)
    
    new_sample_base = np.mean(X_full_scaled, axis=0) 
    new_sample = (new_sample_base * 0.9).reshape(1, -1) 
    
    predicted_class_code = model.predict(new_sample)[0]
    
    if predicted_class_code < len(target_names):
        predicted_class_name = target_names[predicted_class_code]
    else:
         predicted_class_name = f"Class Code {predicted_class_code} (Out of range)"
    
    print("Example Prediction (Handwritten Digit Recognition):")
    print(f"Input Features (First 5 Scaled Pixel Values): {new_sample[0][:5]}") 
    print(f"Predicted Digit: {predicted_class_name}")

# --- Main Execution ---
if __name__ == '__main__':
    # Step 1: Load and Split Data
    X_train, X_test, y_train, y_test, target_names, X_full_scaled = load_and_prepare_data()
    
    # Step 2: Train Model
    svm_model = train_nonlinear_svm(X_train, y_train) 
    
    # Step 3: Evaluate and Predict
    evaluate_and_predict(svm_model, X_test, y_test, target_names, X_full_scaled)