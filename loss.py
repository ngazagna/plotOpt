import numpy as np

def f(beta):
    return (beta[0]-1.)**2 + beta[1]**2 + 1.24
    # return (beta[0]-.2)**2 + (beta[1]-.56)**2 + 1.24

def grad_f(beta):
    return np.array([2*(beta[0]-.2), 2*(beta[1]-.56)])

