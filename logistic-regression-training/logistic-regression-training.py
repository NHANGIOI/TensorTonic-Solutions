import numpy as np
def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
def Binary_Cross_Entropy(pred,y):
    res = 0.0
    n = len(y)
    for i in range(len(y)):
        res += y[i] * np.log(pred[i]) + (1 - y[i]) * np.log(1 - pred[i])
    return - res / n
def train_logistic_regression(X : np.ndarray, y, lr = 0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    n = X.shape[0]
    dimension = X.shape[1]
    weight = np.empty(shape = (dimension,),dtype = float)
    for i in range(weight.shape[0]):
        weight[i] = 0.0
    bias = 0.0
    for step in range(steps):
        z = X @ weight + bias
        pred = _sigmoid(z)
        error = pred - y
        derivative_weight = X.T @ error / n
        derivative_bias = np.mean(error)
        
        weight -= lr * derivative_weight
        bias -= lr * derivative_bias
    return (weight,bias)