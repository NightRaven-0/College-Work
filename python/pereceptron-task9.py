# mlp_numpy.py
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------
# Utilities
# -------------------------
def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, y_true_onehot):
    # mean cross-entropy
    n = probs.shape[0]
    return -np.sum(y_true_onehot * np.log(probs + 1e-12)) / n

def one_hot(y, n_classes):
    oh = np.zeros((y.size, n_classes))
    oh[np.arange(y.size), y] = 1
    return oh

# Activation and derivative for hidden layer
def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(float)

# -------------------------
# MLP class (1 hidden layer)
# -------------------------
class MyMLP():
    def __init__(self, n_in, n_hidden, n_out, lr=0.01, seed=42):
        rng = np.random.RandomState(seed)
        # He init for ReLU
        self.W1 = rng.randn(n_in, n_hidden) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.randn(n_hidden, n_out) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((1, n_out))
        self.lr = lr

    def forward(self, X):
        Z1 = X.dot(self.W1) + self.b1        # (N, n_hidden)
        A1 = relu(Z1)                        # (N, n_hidden)
        Z2 = A1.dot(self.W2) + self.b2       # (N, n_out)
        A2 = softmax(Z2)                     # (N, n_out)
        cache = (X, Z1, A1, Z2, A2)
        return A2, cache

    def backward(self, cache, Y_true_onehot):
        X, Z1, A1, Z2, A2 = cache
        N = X.shape[0]

        # output layer gradient
        dZ2 = (A2 - Y_true_onehot) / N       # (N, n_out)
        dW2 = A1.T.dot(dZ2)                  # (n_hidden, n_out)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # hidden layer gradient
        dA1 = dZ2.dot(self.W2.T)             # (N, n_hidden)
        dZ1 = dA1 * d_relu(Z1)               # (N, n_hidden)
        dW1 = X.T.dot(dZ1)                   # (n_in, n_hidden)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # gradient step
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict_proba(self, X):
        A2, _ = self.forward(X)
        return A2

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# -------------------------
# Demo: train on make_moons
# -------------------------
if __name__ == "__main__":
    # Data
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    n_in = X.shape[1]
    n_hidden = 32
    n_out = len(np.unique(y))
    y_train_oh = one_hot(y_train, n_out)
    y_test_oh = one_hot(y_test, n_out)

    # Model
    mlp = MyMLP(n_in=n_in, n_hidden=n_hidden, n_out=n_out, lr=0.1, seed=1)

    # Training loop
    epochs = 600
    print_every = 100
    for epoch in range(1, epochs + 1):
        probs, cache = mlp.forward(X_train)
        loss = cross_entropy_loss(probs, y_train_oh)
        mlp.backward(cache, y_train_oh)

        if epoch % print_every == 0 or epoch == 1:
            train_pred = mlp.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            print(f"Epoch {epoch:4d}  loss={loss:.4f}  train_acc={train_acc:.4f}")

    # Evaluation
    y_pred = mlp.predict(X_test)
    print("\nTest accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))