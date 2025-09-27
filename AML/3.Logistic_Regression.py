import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer # New import for real dataset
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Load and Prepare Real-World Data ---
def load_and_prepare_data(test_size=0.3, random_state=42):
    """
    Loads the Breast Cancer Wisconsin (Diagnostic) dataset,
    and splits the data into training and testing sets.
    Features: 30 features (e.g., mean radius, texture)
    Target: 0 (malignant) or 1 (benign)
    """
    print("Loading Breast Cancer dataset and preparing for Logistic Regression...")
    
    # Load the real-world dataset (automatically downloaded/cached by scikit-learn)
    data = load_breast_cancer()
    X = data.data
    y = data.target
    target_names = data.target_names 

    # Split the data, ensuring the classes are distributed proportionally (stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Features (Dimension): {X.shape[1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("-" * 50)
    
    # Return the full feature set (X) for creating a realistic sample for prediction later
    return X_train, X_test, y_train, y_test, target_names, X 

# --- 2. Initialize and Train the Logistic Regression Model ---
def train_logistic_regression(X_train, y_train, max_iter=1000):
    """
    Initializes and trains the Logistic Regression classifier.
    """
    print(f"Training Logistic Regression model (max_iter={max_iter})...")
    # Initialize the Logistic Regression classifier
    model = LogisticRegression(max_iter=max_iter, random_state=42)
    # The solver 'liblinear' is generally good for small datasets, but 'lbfgs' is default.
    model.fit(X_train, y_train)
    print("Training complete.")
    return model

# --- 3. Predict, Evaluate, and Demonstrate ---
def evaluate_and_predict(model, X_test, y_test, target_names, X_full):
    """
    Evaluates the model on the test set and makes a sample prediction.
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("-" * 50)
    print("Model Evaluation (Logistic Regression):")
    print(f"Accuracy Score on Test Data: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    # Note: target_names will now be ['malignant' 'benign']
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("-" * 50)
    
    # Demonstrate prediction on a new, unseen data point
    # Use the mean of all features as a representative 'new' sample for prediction
    # We slightly perturb the values (multiply by 1.05) just to make it a new data point
    new_sample_base = np.mean(X_full, axis=0) 
    new_sample = (new_sample_base * 1.05).reshape(1, -1) 
    
    predicted_class_code = model.predict(new_sample)[0]
    predicted_class_name = target_names[predicted_class_code]
    
    # Predict probabilities as well
    predicted_proba = model.predict_proba(new_sample)[0]
    
    print("Example Prediction (Malignant vs. Benign):")
    # Print only the first 5 of 30 features for console readability
    print(f"Input Features (First 5): {new_sample[0][:5]}") 
    print(f"Predicted Class (Code {predicted_class_code}): {predicted_class_name}")
    print(f"Prediction Probabilities (Malignant, Benign): {predicted_proba}")

# --- Main Execution ---
if __name__ == '__main__':
    # Step 1: Load and Split Data
    # X_full is returned to help create a representative sample for prediction
    X_train, X_test, y_train, y_test, target_names, X_full = load_and_prepare_data()
    
    # Step 2: Train Model
    # Setting max_iter to 2000 to ensure convergence on this 30-feature dataset
    logistic_model = train_logistic_regression(X_train, y_train, max_iter=2000) 
    
    # Step 3: Evaluate and Predict
    evaluate_and_predict(logistic_model, X_test, y_test, target_names, X_full)