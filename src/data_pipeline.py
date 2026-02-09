import numpy as np
import gzip
import os

DATA_DIR = "../data"

def load_images(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 784).astype(np.float32) / 255.0

def load_labels(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data
  # one-hot
def one_hot(y, num_classes=10):
      return np.eye(num_classes)[y]

def load_data():
  # paths
  X_train = load_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz"))
  y_train = load_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"))
  X_test  = load_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"))
  y_test  = load_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"))

  y_train = one_hot(y_train)
  y_test  = one_hot(y_test)

  return (X_train, y_train, X_test, y_test)
