import numpy as np

def f(beta):
    x_star = .25
    y_star = 0.75
    f_star = 1.24
    return (beta[0] - x_star)**2 + (beta[1] - y_star)**2 + f_star

def grad_f(beta):
    x_star = .25
    y_star = 0.75
    return np.array([2*(beta[0] - x_star), 2*(beta[1] - y_star)])


# TODO: A collection of optimization test functions Rosenbruck function
