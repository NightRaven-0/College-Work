import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Load and Prepare the Data ---
def load_and_prepare_data(test_size=0.3, random_state=42):
    """
    Loads the Iris dataset, extracts features (X) and target (y),
    and splits the data into training and testing sets.
    """
    print("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data    # Features: [sepal length, sepal width, petal length, petal width]
    y = iris.target  # Targets: 0=setosa, 1=versicolor, 2=virginica
    
    # Split the data, ensuring the classes are distributed proportionally (stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("-" * 30)
    
    return X_train, X_test, y_train, y_test, iris.target_names

# --- 2. Initialize and Train the kNN Model ---
def train_knn(X_train, y_train, k=5):
    """
    Initializes and trains the k-Nearest Neighbors classifier.
    k=5 is a common choice for this small dataset.
    """
    print(f"Training kNN classifier with k={k}...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print("Training complete.")
    return knn

# --- 3. Predict, Evaluate, and Demonstrate ---
def evaluate_and_predict(knn_model, X_test, y_test, target_names):
    """
    Evaluates the model on the test set and makes a sample prediction.
    """
    # Make predictions on the test set
    y_pred = knn_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("-" * 30)
    print(f"Model Evaluation (k={knn_model.n_neighbors}):")
    print(f"Accuracy Score: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("-" * 30)
    
    # Demonstrate prediction on a new, unseen data point
    # Sample data for a likely Iris Setosa: [sepal_l, sepal_w, petal_l, petal_w]
    new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    predicted_species_code = knn_model.predict(new_sample)[0]
    predicted_species_name = target_names[predicted_species_code]
    
    print("Example Prediction:")
    print(f"Input Features: {new_sample[0]}")
    print(f"Predicted Species (Code {predicted_species_code}): {predicted_species_name}")

# --- Main Execution ---
if __name__ == '__main__':
    # Define k, the number of neighbors
    K_VALUE = 5
    
    # Step 1: Load and Split Data
    X_train, X_test, y_train, y_test, target_names = load_and_prepare_data()
    
    # Step 2: Train Model
    knn_model = train_knn(X_train, y_train, k=K_VALUE)
    
    # Step 3: Evaluate and Predict
    evaluate_and_predict(knn_model, X_test, y_test, target_names)