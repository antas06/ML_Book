import numpy as np

class Node:
  def __init__(self, feature =None, threshold = None, left =None, right = None, *, value = None ):
    self.value = value
    self.left= left
    self.right= right
    self.feature= feature
    self.threshold= threshold
  
  def is_leaf_node(self):
    return True if self.value else False


class Decisiontree:
  def __init__(self, min_smaple_split=2, max_depth = 100, n_features = None):
    self.min_sample_split = min_smaple_split
    self.max_depth = max_depth
    self.n_features = n_features
    self.root = None

  

  