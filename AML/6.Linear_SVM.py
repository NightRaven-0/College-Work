import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC 
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Load and Prepare Real-World Data ---
def load_and_prepare_data(test_size=0.3, random_state=42):
    print("Loading Digits dataset and preparing for Linear SVM Classifier...")
    
    # Load the real-world dataset (automatically downloaded/cached by scikit-learn)
    data = load_digits()
    X = data.data
    y = data.target
    # Target names are just the string representations of 0 through 9
    target_names = [str(i) for i in data.target_names] 

    # Split the data, ensuring the classes are distributed proportionally (stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Features (Dimension): {X.shape[1]} (8x8 pixel values)")
    print(f"Target Classes: {len(target_names)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("-" * 50)
    
    return X_train, X_test, y_train, y_test, target_names, X 

# --- 2. Initialize and Train the Linear Support Vector Machine (LinearSVC) ---
def train_linear_svm(X_train, y_train, random_state=42):
    print(f"Training Linear Support Vector Machine (LinearSVC)...")
    model = LinearSVC(random_state=random_state, dual=False, max_iter=20000) 
    model.fit(X_train, y_train)
    print("Training complete.")
    return model

# --- 3. Predict, Evaluate, and Demonstrate ---
def evaluate_and_predict(model, X_test, y_test, target_names, X_full):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("-" * 50)
    print("Model Evaluation (Linear Support Vector Machine / LinearSVC):")
    print(f"Accuracy Score on Test Data: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("-" * 50)

    new_sample_base = np.mean(X_full, axis=0) 
    # Perturb the values slightly for a new data point
    new_sample = (new_sample_base * 1.05).reshape(1, -1) 
    
    predicted_class_code = model.predict(new_sample)[0]
    predicted_class_name = target_names[predicted_class_code]
    
    print("Example Prediction (Digit Classification):")
    print(f"Input Features (First 8 pixels): {new_sample[0][:8]}") 
    print(f"Predicted Class (Code {predicted_class_code}): {predicted_class_name}")

# --- Main Execution ---
if __name__ == '__main__':
    # Step 1: Load and Split Data
    X_train, X_test, y_train, y_test, target_names, X_full = load_and_prepare_data()
    
    # Step 2: Train Model
    svm_model = train_linear_svm(X_train, y_train) 
    
    # Step 3: Evaluate and Predict
    evaluate_and_predict(svm_model, X_test, y_test, target_names, X_full)