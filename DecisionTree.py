import numpy as np

#Creating a class Node
class Node:
  def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.value = value

#Class Decision Tree
class DecisionTree:
  #init the props of dt.
  def __init__(self, max_depth=10, n_features=None):
    self.max_depth = max_depth
    self.n_features = n_features
    self.root = None

  #Growing tree from root
  def fit(self, X,y):
    self.root = self._grow_tree(X,y, depth =0)

  def _grow_tree(self, X, y, depth):
    #Stopping con
    if depth >= self.max_depth or len(set(y)) == 1 or len(y) < 2:
      # Leaf node =majority class
      leaf_value = max(set(y), key=list(y).count)
      return Node(value=leaf_value)

    n_samples, n_feats = len(X), len(X[0])
    best_gain = 0
    best_feature = 0
    best_threshold = X[0][0]

    for feature in range(n_feats):
      values = set([row[feature] for row in X])
      for threshold in values:
        left = [y[i] for i in range(n_samples) if X[i][feature] <= threshold]
        right = [y[i] for i in range(n_samples) if X[i][feature] > threshold]
        if not left or not right:
          continue
        gain = abs(len(left) - len(right))
        if gain > best_gain:
          best_gain = gain
          best_feature = feature
          best_threshold = threshold

    left_X = [X[i] for i in range(n_samples) if X[i][best_feature] <= best_threshold]
    left_y = [y[i] for i in range(n_samples) if X[i][best_feature] <= best_threshold]
    right_X = [X[i] for i in range(n_samples) if X[i][best_feature] > best_threshold]
    right_y = [y[i] for i in range(n_samples) if X[i][best_feature] > best_threshold]

    left_node = self._grow_tree(left_X, left_y, depth + 1)
    right_node = self._grow_tree(right_X, right_y, depth + 1)

    return Node(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)


  def predict(self, X):
    return [self._predict_row(row, self.root) for row in X]

  def _predict_row(self, row, node):
    if node.value is not None:
      return node.value
    if row[node.feature] <= node.threshold:
      return self._predict_row(row, node.left)
    else:
      return self._predict_row(row, node.right)