import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB # Naive Bayes Classifier
from sklearn.datasets import load_wine # Changed dataset to Wine
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Load and Prepare Real-World Data ---
def load_and_prepare_data(test_size=0.3, random_state=42):
    print("Loading Wine dataset and preparing for Naive Bayesian Classifier...")
    
    # Load the real-world dataset (automatically downloaded/cached by scikit-learn)
    data = load_wine()
    X = data.data
    y = data.target
    target_names = data.target_names 

    # Split the data, ensuring the classes are distributed proportionally (stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Features (Dimension): {X.shape[1]} (e.g., Alcohol, Malic Acid)")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("-" * 50)
    
    # Return the full feature set (X) for creating a realistic sample for prediction later
    return X_train, X_test, y_train, y_test, target_names, X 

# --- 2. Initialize and Train the Naive Bayesian Classifier (GaussianNB) ---
def train_naive_bayes(X_train, y_train):
    print(f"Training Naive Bayesian Classifier (GaussianNB)...")
    # Initialize the Gaussian Naive Bayes classifier
    model = GaussianNB() 
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
    print("Model Evaluation (Naive Bayesian Classifier / GaussianNB):")
    print(f"Accuracy Score on Test Data: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    # target_names are the names of the three wine classes
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("-" * 50)
    
    # Demonstrate prediction on a new, unseen data point
    # Use the mean of all features as a representative 'new' sample for prediction
    new_sample_base = np.mean(X_full, axis=0) 
    # Perturb the values slightly for a new data point
    new_sample = (new_sample_base * 1.05).reshape(1, -1) 
    
    predicted_class_code = model.predict(new_sample)[0]
    predicted_class_name = target_names[predicted_class_code]
    
    # Predict probabilities (the probability of each class)
    predicted_proba = model.predict_proba(new_sample)[0]
    
    print("Example Prediction (Wine Classification):")
    # Wine dataset has 13 features; print the first 5 for readability
    print(f"Input Features (First 5): {new_sample[0][:5]}") 
    print(f"Predicted Class (Code {predicted_class_code}): {predicted_class_name}")
    print(f"Prediction Probabilities (Class 0, Class 1, Class 2): {predicted_proba}")

# --- Main Execution ---
if __name__ == '__main__':
    # Step 1: Load and Split Data
    X_train, X_test, y_train, y_test, target_names, X_full = load_and_prepare_data()
    
    # Step 2: Train Model
    nb_model = train_naive_bayes(X_train, y_train) 
    
    # Step 3: Evaluate and Predict
    evaluate_and_predict(nb_model, X_test, y_test, target_names, X_full)