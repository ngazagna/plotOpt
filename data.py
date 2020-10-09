import numpy as np

def generate_data(n_samples, n_features):
    X = np.random.randn(n_samples, n_features)
    beta = np.random.randn(n_features)
    y = np.sign(X.dot(beta))
    return X, y