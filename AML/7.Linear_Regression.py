import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # Changed model to Linear Regression
from sklearn.datasets import load_diabetes # Changed dataset to Diabetes (Regression)
from sklearn.metrics import mean_squared_error, r2_score # Changed metrics to Regression metrics

# --- 1. Load and Prepare Real-World Data ---
def load_and_prepare_data(test_size=0.3, random_state=42):
    print("Loading Diabetes dataset and preparing for Linear Regression...")
    
    data = load_diabetes()
    X = data.data
    y = data.target
    target_names = ["Disease Progression"]

    # Split the data. Stratification is not applied in regression.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Features (Dimension): {X.shape[1]} (e.g., BMI, Blood Pressure)")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("-" * 50)

    return X_train, X_test, y_train, y_test, target_names, X 

# --- 2. Initialize and Train the Linear Regression Model ---
def train_linear_regression(X_train, y_train):
    print(f"Training Linear Regression Model...")
    # Initialize the Linear Regression model
    model = LinearRegression() 
    model.fit(X_train, y_train)
    print("Training complete.")
    return model

# --- 3. Predict, Evaluate, and Demonstrate ---
def evaluate_and_predict(model, X_test, y_test, target_names, X_full):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("-" * 50)
    print("Model Evaluation (Linear Regression):")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    print(f"Ideal R2 score is 1.0, lower MSE is better.")
    print("-" * 50)
    
    new_sample_base = np.mean(X_full, axis=0) 
    new_sample = (new_sample_base * 0.9).reshape(1, -1) 
    
    predicted_value = model.predict(new_sample)[0]
    
    print("Example Prediction (Diabetes Progression):")
    print(f"Input Features (First 5): {new_sample[0][:5]}") 
    print(f"Predicted Disease Progression: {predicted_value:.2f}")

# --- Main Execution ---
if __name__ == '__main__':
    # Step 1: Load and Split Data
    X_train, X_test, y_train, y_test, target_names, X_full = load_and_prepare_data()
    
    # Step 2: Train Model
    reg_model = train_linear_regression(X_train, y_train) 
    
    # Step 3: Evaluate and Predict
    evaluate_and_predict(reg_model, X_test, y_test, target_names, X_full)