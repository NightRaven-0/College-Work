#23BAI11010 RF
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Dummy dataset
X, y = make_classification(
    n_samples=300,
    n_features=10,
    n_informative=6,
    n_classes=3,
    random_state=42
)

# Base model: Random Forest
rf = RandomForestClassifier(n_estimators=50, random_state=42)

# Bagging with Random Forest (use 'estimator' for new sklearn versions)
bagging_clf = BaggingClassifier(
    estimator=rf,
    n_estimators=10,
    random_state=42
)

# Train
bagging_clf.fit(X, y)

# Predict
y_pred = bagging_clf.predict(X)

# Evaluate
print("Training Accuracy:", accuracy_score(y, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))

# Show how many estimators in bagging
print("Number of base estimators in Bagging:", len(bagging_clf.estimators_))