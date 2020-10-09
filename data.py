import numpy as np

def generate_binary(n_samples, n_features):
    """
    Generate binary classification data with dense design matrix.
    Parameters
    ----------
    n_samples : int
        Number of data samples.
    n_features : int
        Number of data samples.
    Returns
    ----------
    X : ndarray of shape (n_samples, n_features)
        Randomly sampled design matrix.
    y : ndarray of shape (n_samples, 1)
        Randomly labels sampled in {-1, 1}.
    """
    X = np.random.randn(n_samples, n_features)
    beta = np.random.randn(n_features)
    y = np.sign(X.dot(beta))
    return X, y