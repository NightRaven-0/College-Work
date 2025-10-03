import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.datasets import fetch_covtype 
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.preprocessing import StandardScaler 

# --- 1. Load and Prepare Real-World Data ---
def load_and_prepare_data(test_size=0.3, random_state=42):
    print("Loading Forest Covertype dataset and preparing for AdaBoost Classification...")
    
    # Load the dataset (shuffle is set to True by default, taking a subset)
    data = fetch_covtype(as_frame=False, shuffle=True, random_state=42)
    X = data.data
    y = data.target
    

    N_SAMPLES = 10000 
    X = X[:N_SAMPLES]
    y = y[:N_SAMPLES]
    
    unique_y = np.unique(y)
    target_names = [f"Cover_Type_{i}" for i in unique_y]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_scaled_full = scaler.transform(X)

    print(f"Total samples used: {len(X)}")
    print(f"Features (Dimension): {X.shape[1]} (Elevation, Soil Type etc.)")
    print(f"Target Classes: {len(target_names)} (Cover Types 1 through 7)")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("-" * 50)
    
    # Return the scaled data
    return X_train, X_test, y_train, y_test, target_names, X_scaled_full 

# --- 2. Initialize and Train the AdaBoost Classifier ---
def train_adaboost_classifier(X_train, y_train): 
    print(f"Training AdaBoost Classifier (using Decision Tree stumps)...")
    
    # 1. Define the base estimator (shallow tree/stump)
    base_estimator = DecisionTreeClassifier(max_depth=2, random_state=42) 
    
    # 2. Initialize the AdaBoost classifier 
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=50, 
        learning_rate=1.0, 
        random_state=42,
        algorithm='SAMME' 
    ) 
    
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
    print("Model Evaluation (AdaBoost Classifier):")
    print(f"Accuracy Score on Test Data: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    print("-" * 50)

    new_sample_base = np.mean(X_full_scaled, axis=0) 

    new_sample = (new_sample_base * 0.9).reshape(1, -1) 
    
    predicted_class_code = model.predict(new_sample)[0]
    predicted_class_name = f"Cover_Type_{predicted_class_code}"
    
    print("Example Prediction (Forest Covertype Classification):")
    print(f"Input Features (First 5 Scaled): {new_sample[0][:5]}") 
    print(f"Predicted Cover Type: {predicted_class_name} (Code: {predicted_class_code})")

# --- Main Execution ---
if __name__ == '__main__':
    # Step 1: Load and Split Data
    X_train, X_test, y_train, y_test, target_names, X_full_scaled = load_and_prepare_data()
    
    # Step 2: Train Model
    adaboost_model = train_adaboost_classifier(X_train, y_train) 
    
    # Step 3: Evaluate and Predict
    evaluate_and_predict(adaboost_model, X_test, y_test, target_names, X_full_scaled)