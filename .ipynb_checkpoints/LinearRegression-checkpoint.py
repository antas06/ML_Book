class LinearRegression:
  def __init__(self, learning_rate=0.01, epochs=1000):
    self.learning_rate = learning_rate
    self.epoch = epochs
    self.w= None
    self.b= None

  def mean(self, values):
    return sum(values)/len(values)
  
  def mse(self, y_pred, y_true):
    return sum(((y_pred[i]-y_true[i]) **2) for i in range(len(y_true)))/ len(y_true)
  
  def predict(self, X):
    y_pred = []
    for i in range(len(X)):
      y_hat = sum(self.w[j] * X[i][j] for j in range(len(X[0]))) + self.b
      y_pred.append(y_hat)
    return y_pred

  def gradient_descent(self, X, y):
    n_samples, n_features = len(X), len(X[0])

    self.w = [0.0 for ev in range(n_features)]
    self.b = 0.0

    for ep in range(self.epoch):
      y_pred = self.predict(X)

      dw = [0.0 for fe in range(n_features)] #features ke saath h
      db = 0.0

      for i in range(n_samples):
        error = y_pred[i] - y[i]
        for j in range(n_features):
          dw[j] += (2/n_samples) * error * X[i][j]  #yeh derivative h
        db += (2/n_samples) * error

      for j in range(n_features):
        self.w[j] -= self.learning_rate * dw[j]
      self.b -= self.learning_rate * db

    return self.w, self.b
  
  def fit(self,X,y):
    self.w, self.b = self.gradient_descent(X, y)
    return self.w, self.b

  def fet_parms(self):
    return {"Weights": self.w, "Bias": self.b}
  
  def score_mse(self, X, y):
    y_pred = self.predict(X)
    return self.mse(y_pred, y)