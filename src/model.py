import numpy as np

class MLP_Classifier:
    def __init__(self, dims, lr=0.01, epochs=40, batch_size=64):
        # dims = [in, h1, h2, ..., out]
        self.dims = dims
        self.lr = lr
        self.epochs = epochs
        self.batch = batch_size
        self._init_params()

    # ---------- init ----------
    def _init_params(self):
        self.W = []
        self.b = []
        for i in range(len(self.dims)-1):
            self.W.append(np.random.randn(self.dims[i], self.dims[i+1]) * 0.01)
            self.b.append(np.zeros((1, self.dims[i+1])))

    # ---------- activations ----------
    def relu(self, Z): 
        return np.maximum(0, Z)

    def drelu(self, Z): 
        return (Z > 0).astype(np.float32)

    def softmax(self, Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        eZ = np.exp(Z)
        return eZ / np.sum(eZ, axis=1, keepdims=True)

    # ---------- loss ----------
    def cross_entropy(self, y_hat, y):
        eps = 1e-12
        return -np.mean(np.sum(y * np.log(y_hat + eps), axis=1))

    # ---------- forward ----------
    def forward(self, X):
        self.Z = []
        self.A = [X]

        for i in range(len(self.W)-1):
            Z = self.A[-1] @ self.W[i] + self.b[i]
            A = self.relu(Z)
            self.Z.append(Z)
            self.A.append(A)

        # output layer
        Z = self.A[-1] @ self.W[-1] + self.b[-1]
        A = self.softmax(Z)
        self.Z.append(Z)
        self.A.append(A)

        return A

    # ---------- backward ----------
    def backward(self, y):
        m = y.shape[0]
        dZ = (self.A[-1] - y) / m   # softmax+CE shortcut

        for i in reversed(range(len(self.W))):
            dW = self.A[i].T @ dZ
            db = np.sum(dZ, axis=0, keepdims=True)

            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db

            if i > 0:
                dA = dZ @ self.W[i].T
                dZ = dA * self.drelu(self.Z[i-1])

    # ---------- train ----------
    def fit(self, X, y):
        for ep in range(self.epochs):
            perm = np.random.permutation(X.shape[0])
            Xs, ys = X[perm], y[perm]

            for i in range(0, X.shape[0], self.batch):
                Xb = Xs[i:i+self.batch]
                yb = ys[i:i+self.batch]

                y_hat = self.forward(Xb)
                L = self.cross_entropy(y_hat, yb)
                self.backward(yb)

            print(ep, L)

    # ---------- predict ----------
    def predict(self, X):
        y_hat = self.forward(X)
        return np.argmax(y_hat, axis=1)
