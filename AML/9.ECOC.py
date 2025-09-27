#23BAI11010 ECOC
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OutputCodeClassifier
from sklearn.metrics import accuracy_score

# Create a dummy multi-class dataset
X, y = make_classification(
    n_samples=300,     # number of samples
    n_features=10,     # number of features
    n_informative=6,   # informative features
    n_classes=4,       # number of classes (ECOC useful here!)
    random_state=42
)

# ECOC with Logistic Regression as base classifier
base_clf = LogisticRegression(max_iter=1000)
ecoc_clf = OutputCodeClassifier(base_clf, code_size=2.0, random_state=42)

# Train
ecoc_clf.fit(X, y)

# Predict
y_pred = ecoc_clf.predict(X)

# Evaluate
print("Training Accuracy:", accuracy_score(y, y_pred))
print("Classes learned:", ecoc_clf.classes_)