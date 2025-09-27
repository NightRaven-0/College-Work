#23BAI11010 perceptron
import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=20):
        # weights[0] is bias, weights[1:] are feature weights
        self.weights = np.zeros(input_size + 1, dtype=float)
        self.lr = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict_raw(self, x):
        # x: 1D feature array
        return np.dot(self.weights[1:], x) + self.weights[0]

    def predict(self, x):
        x = np.atleast_2d(x)
        outs = [self.activation(self.predict_raw(xi)) for xi in x]
        return np.array(outs)

    def train(self, X, y):
        # X shape: (n_samples, n_features)
        for epoch in range(self.epochs):
            for xi, target in zip(X, y):
                pred = self.predict(xi)[0]
                err = target - pred
                self.weights[1:] += self.lr * err * xi
                self.weights[0] += self.lr * err


def confusion_matrix_and_metrics(y_true, y_pred):
    # Labels: 0 = Even, 1 = Odd
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    accuracy = (tp + tn) / cm.sum()
    precision_even = tn / (tn + fn) if (tn + fn) > 0 else 0.0 
    recall_even = tn / (tn + fp) if (tn + fp) > 0 else 0.0 
    precision_odd = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_odd = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "precision_even": precision_even,
        "recall_even": recall_even,
        "precision_odd": precision_odd,
        "recall_odd": recall_odd,
        "counts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return metrics

if __name__ == "__main__":
    # Prepare data: ASCII codes for digits
    digits = np.arange(10)
    ascii_codes = np.array([ord(str(d)) for d in digits], dtype=int)

    # Feature: least-significant-bit of ASCII code -> parity bit (0 even, 1 odd)
    X = (ascii_codes & 1).reshape(-1, 1).astype(float)

    # Labels: 0 for even digit, 1 for odd digit
    y = np.array([0 if d % 2 == 0 else 1 for d in digits])

    # Instantiate and train perceptron
    percep = Perceptron(input_size=1, learning_rate=0.2, epochs=10)
    percep.train(X, y)

    # Predict on same dataset because there's only 10 samples and no noise
    y_pred = percep.predict(X)

    # Print per-digit predictions
    print("Per-digit predictions:")
    for d, asc, feat, pred in zip(digits, ascii_codes, X.flatten(), y_pred):
        print(f"Digit {d}  ASCII {asc}  LSB {int(feat)}  -> Predicted: {'Odd' if pred else 'Even'}")

    m = confusion_matrix_and_metrics(y, y_pred)

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("         Pred_Even  Pred_Odd")
    print(f"True_Even    {m['confusion_matrix'][0,0]:>3}         {m['confusion_matrix'][0,1]:>3}")
    print(f"True_Odd     {m['confusion_matrix'][1,0]:>3}         {m['confusion_matrix'][1,1]:>3}")

    print(f"\nAccuracy: {m['accuracy']*100:.2f}%")
    print("\nPer-class metrics:")
    print(f"Even  - Precision: {m['precision_even']:.2f}, Recall: {m['recall_even']:.2f}")
    print(f"Odd   - Precision: {m['precision_odd']:.2f}, Recall: {m['recall_odd']:.2f}")
    print("\nRaw counts:", m["counts"])