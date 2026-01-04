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
  def __init__(self, max_depth=5, n_features=None):
    self.max_depth = max_depth
    self.n_features = n_features
    self.root = None

#Growing tree from root
  def fit(self, X,y):
    self.root = self._grow_tree(X,y, depth =0)

  

    



  