import numpy as np
from collections import Counter

def distance(x1, x2):
  return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
  def __init__(self, k=3):
       self.k = k

  def fit(self, X, y):
    self.X_train, self.y_train = X, y

  def predict(self, X):
    return [self._predict(x) for x in X]

  def _predict(self, x):
    # Compute distances
    distances = [distance(x, x_train) for x_train in self.X_train]
        
    # Get k nearest labels
    k_indices = np.argsort(distances)[:self.k]
    k_labels = [self.y_train[i] for i in k_indices]
        
    # Majority vote
    return Counter(k_labels).most_common(1)[0][0]