import numpy as np
from scipy.optimize import check_grad
from abc import ABC, abstractmethod


class Problem(ABC):
    """Abstract class defining a problem"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    @abstractmethod
    def loss(self, beta):
        """Loss function of the problem at beta

        Args:
            beta : weights vector
        """
        pass

    @abstractmethod
    def grad(self, beta):
        """Gradient function of the loss at beta

        Args:
            beta : weights vector
        """
        pass


class LogregL2Problem(Problem):
    def __init__(self, X, y, lmbd):
        super().__init__(X, y)
        self.lmbd = lmbd

    def loss(self, beta):
        return loss_logreg_l2(beta, self.X, self.y, self.lmbd)

    def grad(problem, beta):
        return grad_logreg_l2(beta, self.X, self.y, self.lmbd)


# def sigmoid_scalar(x):
#     "Numerically stable sigmoid function."
#     if x >= 0:
#         z = np.exp(-x)
#         return 1. / (1. + z)
#     else:
#         # if x is less than zero then z will be small, denom can't be
#         # zero because it's 1+z.
#         z = np.exp(x)
#         return z / (1. + z)

# def sigmoid(x):
#     return np.array(list(map(sigmoid_scalar, x)))

def sigmoid(t):
    """Sigmoid function"""
    return 1. / (1. + np.exp(-t))

def loss_logreg_l2(beta, X, y, lmbd):
    n_samples = X.shape[0]
    y_X_beta = y * X.dot(beta.flatten())
    l2 = 0.5 * np.dot(beta, beta)
    # return (np.sum(np.log(1 + np.exp(-y_X_beta))) / n_samples) + lmbd * l2
    return (1/n_samples) * np.log1p(np.exp(-y_X_beta)).sum() + lmbd * l2

def grad_logreg_l2(beta, X, y, lmbd):
    n_samples = X.shape[0]
    y_X_beta = y * X.dot(beta.flatten())
    return (X.T.dot(y * (sigmoid(y_X_beta) - 1)) / n_samples) + lmbd * beta


if __name__ == "__main__":
    np.random.seed(0)

    n_samples, n_features = 1000, 20
    X, y = generate_data(n_samples, n_features)
    lmbd = .1
    problem = LogregL2Problem(X, y, lmbd)

    beta = np.random.randn(n_features)

    grad_logreg_l2(beta, X, y, lmbd)

    a = check_grad(loss_logreg_l2, grad_logreg_l2, beta, X, y, lmbd)

