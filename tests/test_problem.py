import numpy as np
import pytest
from scipy.optimize import check_grad

from sandbox import LogregL2Problem, loss_logreg_l2, grad_logreg_l2

@pytest.mark.problem
class TestProblem:
    def test_init_logreg_l2(self, dataset, lmbd):
        """Confirms l2 regularized logistic regression
        problem can be instantiated"""
        X, y = dataset
        _ = LogregL2Problem(X, y, lmbd)

    def test_grad_logreg_l2(self, dataset, lmbd):
        X, y = dataset
        tolerance = 1e-6
        n_features = X.shape[1]
        problem = LogregL2Problem(X, y, lmbd)
        num_runs = 100
        avg_error = 0
        for i in range(num_runs):
            beta = np.random.randn(n_features)
            avg_error += check_grad(loss_logreg_l2, grad_logreg_l2, beta, X, y, lmbd)
        avg_error /= num_runs
        print("Average gradient error: ", avg_error)
        assert avg_error < tolerance
