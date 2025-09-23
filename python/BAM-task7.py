import numpy as np

def bipolar(v):
    """Convert 0/1 vectors to bipolar -1/1 (if already -1/1 it returns unchanged)."""
    v = np.array(v)
    # if values are 0/1, convert to -1/1
    if np.any(v == 0) and not np.any(v == -1):
        return np.where(v == 0, -1, 1)
    return v.astype(int)

def sign_with_zero_to_prev(vec, prev=None):
    s = np.sign(vec)
    if prev is None:
        s[s == 0] = 1
    else:
        zeros = (s == 0)
        s[zeros] = prev[zeros]
    return s.astype(int)

class BAM:
    def _init_(self):
        self.W = None  # weight matrix from X to Y

    def train(self, X_list, Y_list):
        """
        X_list: list of 1D arrays (length n_x)
        Y_list: list of 1D arrays (length n_y)
        All vectors should be bipolar (-1,1)
        Weight matrix W = sum_k x_k (y_k)^T
        """
        assert len(X_list) == len(Y_list), "Number of X and Y patterns must match"
        X0 = np.array(X_list[0])
        Y0 = np.array(Y_list[0])
        n_x = X0.size
        n_y = Y0.size
        self.W = np.zeros((n_x, n_y), dtype=int)
        for x, y in zip(X_list, Y_list):
            x = np.array(x).reshape(n_x, 1)
            y = np.array(y).reshape(1, n_y)
            self.W += x @ y  # outer product
        # W is integer matrix
        return self.W

    def recall(self, x_init=None, y_init=None, max_iters=20, verbose=False):
        """
        Perform synchronous alternating updates until convergence or max_iters.
        Provide either x_init (to recall Y) or y_init (to recall X), or both.
        Returns (x_final, y_final)
        """
        assert self.W is not None, "BAM not trained yet"
        n_x, n_y = self.W.shape

        # initialize
        if x_init is None and y_init is None:
            raise ValueError("Provide at least x_init or y_init")
        if x_init is None:
            x = np.random.choice([-1, 1], size=n_x)
        else:
            x = bipolar(x_init).astype(int)
        if y_init is None:
            y = np.random.choice([-1, 1], size=n_y)
        else:
            y = bipolar(y_init).astype(int)

        prev_x, prev_y = None, None
        for i in range(max_iters):
            # update Y from X: y' = sign(W^T x)
            y_in = x @ self.W  # shape (n_y,)
            y_new = sign_with_zero_to_prev(y_in, prev=y)
            # update X from Y: x' = sign(W y)
            x_in = self.W @ y_new  # shape (n_x,)
            x_new = sign_with_zero_to_prev(x_in, prev=x)

            if verbose:
                print(f"Iter {i+1}: x={x_new}, y={y_new}")

            # convergence?
            if np.array_equal(x_new, x) and np.array_equal(y_new, y):
                return x_new, y_new

            prev_x, prev_y = x, y
            x, y = x_new, y_new

        # return last state
        return x, y
 #Example usage
if __name__ == "__main__":
    # define two bipolar pairs (X_i <-> Y_i)
    X1 = [1, -1, 1, -1]    # pattern A (length 4)
    Y1 = [1,  1, -1]       # associated pattern (length 3)

    X2 = [-1, 1, -1, 1]    # pattern B
    Y2 = [-1, -1, 1]       # associated pattern

    # convert to bipolar explicitly (optional)
    X_list = [bipolar(X1), bipolar(X2)]
    Y_list = [bipolar(Y1), bipolar(Y2)]

    bam = BAM()
    W = bam.train(X_list, Y_list)
    print("Weight matrix W (X->Y):\n", W)

    # Perfect recall: X1 -> Y1
    x_test = X1
    x_final, y_final = bam.recall(x_init=x_test, max_iters=10, verbose=False)
    print("\nRecall from X1:")
    print(" Input X:", x_test)
    print(" Recalled Y:", y_final)

    # Perfect recall: Y2 -> X2
    y_test = Y2
    x_final2, y_final2 = bam.recall(y_init=y_test, max_iters=10, verbose=False)
    print("\nRecall from Y2:")
    print(" Input Y:", y_test)
    print(" Recalled X:", x_final2)

    # Noisy recall example: flip one bit in X1
    noisy_x1 = np.array(X1).copy()
    noisy_x1[0] *= -1 
    xn, yn = bam.recall(x_init=noisy_x1, max_iters=10, verbose=True)
    print("\nNoisy recall from flipped X1:")
    print(" Noisy X:", noisy_x1)
    print(" Recalled Y:", yn)
    print(" Final X after convergence:", xn)