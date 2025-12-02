import numpy as np

def Sigmoid(x):
  return 1/(1+np.exp(-x))

class LogisticRegression:
  def __init__(self, learning_rate= 0.01, epoch = 1000):
    self.learning_rate = learning_rate
    self.epoch = epoch
    self.w = None
    self.b = None

  def predict(self, X):
    pow_pred = np.dot(X, self.w) + self.b
    return Sigmoid(pow_pred)
    
  def gradient_descent(self, X, y):
    rows , features = X.shape

    self.w = np.zeros(features)
    self.b = 0.0

    for _ in range (self.epoch):
      preds = self.predict(X)

      dw = (1/rows)*np.dot(X.T, (preds-y))
      db = (1/rows)*np.sum(preds-y)

      self.w = self.w - self.learning_rate * dw
      self.b = self.b - self.learning_rate * db

    return self.w, self.b
  
  def fit(self, X, y):
    self.w, self.b = self.gradient_descent(X,y)
    print("Ho gya Model fit.")
    return {"Weights" : self.w, "Bias" : self.b}
  
  def find_classification(self, X):
    y_curv_pred = self.predict(X)
    self.rows = X.shape[0]
    class_pred = [0 if y_curv_pred <=0.5 else 1 for row in self.rows]
    return class_pred
  
  def accuracy(self, X, y):
    class_pred = self.find_classification(X)
    return (np.sum(np.array(class_pred)==y))/ len(y)