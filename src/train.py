from data_pipeline import load_data
from model import MLP_Classifier
import matplotlib.pyplot as plt
import numpy as np

X_train, y_train, X_test, y_test = load_data()

def accuracy(y_hat, y):
    y_pred = np.argmax(y_hat, axis=1)
    y_true = np.argmax(y, axis=1)
    return np.mean(y_pred == y_true)

model = MLP_Classifier(
    dims=[784, 128, 64, 10],
    lr=0.01,
    epochs=40,
    batch_size=64
)

model.fit(X_train, y_train)

idx = np.random.choice(X_test.shape[0], 64, replace=False)
Xb = X_test[idx]
yb = y_test[idx]

pred = model.predict(Xb)

fig, axes = plt.subplots(8, 8, figsize=(10,10))

for i, ax in enumerate(axes.flat):
    ax.imshow(Xb[i].reshape(28,28), cmap="gray")
    ax.set_title(f"P:{pred[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()