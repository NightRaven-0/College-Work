# delta_sigmoid.py
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def add_bias(X):
    return np.hstack([np.ones((X.shape[0],1)), X])

def predict_sigmoid(W, X):
    z = X.dot(W)
    return sigmoid(z)

def train_delta_sigmoid(X, y, lr=0.1, epochs=200):
    # y must be 0/1
    n, m = X.shape
    W = np.zeros(m)
    for ep in range(epochs):
        z = X.dot(W)
        y_hat = sigmoid(z)
        # MSE loss L = 1/N sum (y - y_hat)^2
        # gradient wrt weights: dL/dW = -2/N * X^T * (y - y_hat) * y_hat*(1-y_hat)
        delta = (y - y_hat) * (y_hat * (1 - y_hat))   # shape (N,)
        grad = - (X.T @ delta) * (2.0 / n)            # negative because (y-y_hat) used
        W = W - lr * grad
        if (ep+1) % (epochs//5) == 0 or ep==0:
            loss = mean_squared_error(y, y_hat)
            print(f"Epoch {ep+1:4d}  MSE={loss:.6f}")
    return W

if __name__ == "__main__":
    # make binary dataset (0/1)
    X, y = make_blobs(n_samples=500, centers=2, n_features=2, random_state=0, cluster_std=1.4)
    # map labels already 0/1 from make_blobs
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    Xb = add_bias(X_train)
    W = train_delta_sigmoid(Xb, y_train.astype(float), lr=0.25, epochs=400)

    Xb_test = add_bias(X_test)
    probs = predict_sigmoid(W, Xb_test)
    preds = (probs >= 0.5).astype(int)
    print("Test accuracy:", accuracy_score(y_test, preds))
    print("Example probs:", probs[:8])