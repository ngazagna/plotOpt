from numba import jit, njit


# @njit
@jit(nopython=False)
def gd(x_init, grad, n_iter=100, step=1., store_every=1, args=()):
    """Gradient descent algorithm."""
    x = x_init.copy()
    x_list = []
    x_list.append(x.copy())
    time_list = []
    time_elapsed = 0.;
    time_list.append(time_elapsed)
    for i in range(n_iter):
        ### GD
        t0 = time()
        x -= step * grad(x, *args)
        time_elapsed += time() - t0
        ### END OF GD
        if i % store_every == 0:
            x_list.append(x.copy())
            time_list.append(time_elapsed)
    return x, x_list, time_list