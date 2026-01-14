import numpy as np

class NavieBayes:

  def fit(self, X, y):
    self.n_classes = np.unique(y)
    self.mean = {}
    self.var = {}
    self.c_prob ={}

    for c in self.n_classes:
      X_c = X[y==c]
      self.mean[c] = np.mean(X_c)
      self.var[c] = np.var(X_c)
      self.c_prob[c] = X_c.shape[0]/X.shape[0]

  # Gaussian Formula, (woh wala curve, chotlundu)
  def gaussian(self, x, mean, var):
    return np.exp(-(x - mean) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
  
  def predict(self, X):
    predictions = []

    for x in X:
      posteriors = []

      for c in self.n_classes:
        prior = np.log(self.c_prob[c])
        likelihood = np.sum(np.log(self.gaussian(x, self.mean[c], self.var[c])))
        posterior = prior + likelihood
        posteriors.append(posterior)

      predictions.append(self.classes[np.argmax(posteriors)])

    return np.array(predictions)
  
  #score for evaluation
  def score(self, X, y):
    predictions = self.predict(X)
    return np.mean(predictions == y)