#23BAI1101 LSVM
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

print("Step 1: Loading pre-split and pre-scaled datasets...")
# Set the base directory for the R drive
base_dir = "R:\\VS CODE\\Dataset"

# Load the four pre-split and pre-scaled data files
X_train_scaled = pd.read_csv(os.path.join("R:\VS CODE\Dataset\X_train_scaled_linear.csv"))
X_test_scaled = pd.read_csv(os.path.join(base_dir, "X_test_scaled_linear.csv"))
y_train = pd.read_csv(os.path.join(base_dir, "y_train_linear.csv")).squeeze()
y_test = pd.read_csv(os.path.join(base_dir, "y_test_linear.csv")).squeeze()

print("\nStep 2: Training a Linear SVM model...")
# Train a Linear SVM model
model = LinearSVC(C=1.0, class_weight="balanced", dual=False)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

print("\nStep 3: Evaluating model performance...")
# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nStep 4: Saving model and feature list...")
# Get the scaler from the pre-scaling step (if you saved it) or re-create it
# If you didn't save the scaler, you'll need to re-fit it to save it
scaler = StandardScaler()
scaler.fit(pd.read_csv(os.path.join(base_dir, "X_train_linear.csv")))

# Save the model with a new name
with open(os.path.join(base_dir, "linearmodel_SVM.pkl"), "wb") as f:
    pickle.dump(model, f)

# Save the scaler with a new name
with open(os.path.join(base_dir, "linearscaler_SVM.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Save the feature list with a new name
with open(os.path.join(base_dir, "linearfeature_list_SVM.txt"), "w") as f:
    for col in X_train_scaled.columns:
        f.write(f"{col}\n")

print("\nLinear SVM model, scaler, and feature list saved in the Dataset folder.")
print("Training complete.")