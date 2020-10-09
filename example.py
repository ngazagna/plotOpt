# Solve a 2D logistic regression problem to generate a dataframe
# with columns: iterations, epochs, time and weights

# from time import time


# input
# >>>> plot2D conv_2D.csv


# Example of `input.csv' for n_samples = 100
# with gradient descent (so 1 epoch per iteration)
# iteration, epoch, time, beta0, beta1
# 0, 0, .0, 1., 1.
# 1, 0, .2, 1., .5
# 2, 0, .6, 1., .0
# 3, 0, 1.2, .0, .0


# Problem: problem.loss function is required...



# if __name__ == "__main__":
import numpy as np
import pandas as pd
import os


df2D = pd.DataFrame(np.array([[0, 0, .0, 1., 1.],
                            [1, 100, .2, 1., .5],
                            [2, 200, .6, 1., .0],
                            [3, 300, 1.2, .0, .0]]),
                            columns=["iteration", "epoch", "time", "beta0", "beta1"])
df2D = df2D.astype({"iteration": int,
                "epoch": float,
                "time": float,
                "beta0": float,
                "beta1": float})
df2D

folder = os.path.join(os.getcwd(), "conv_tables")
full_path = os.path.join(folder, "conv_2D.csv")
df2D.to_csv (full_path, index=False, header=True)



df3D = pd.DataFrame(np.array([[0, 0, .0, 1., 1., 1.],
                            [1, 100, .2, 1., .5, 1.],
                            [2, 200, .6, 1., .0, 1.],
                            [3, 300, 1.2, .0, .0, .5],
                            [4, 400, 2.4, .0, .0, .0]]),
                            columns=["iteration", "epoch", "time", "beta0", "beta1", "beta2"])
df3D = df3D.astype({"iteration": int,
                "epoch": float,
                "time": float,
                "beta0": float,
                "beta1": float})
df3D

folder = os.path.join(os.getcwd(), "conv_tables")
full_path = os.path.join(folder, "conv_3D.csv")
df3D.to_csv (full_path, index=False, header=True)



###################################
########## NOTEBOOK CODE ##########
###################################
import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab


def loss_2_args(x, y):
    return loss(np.array([x, y]), A, b, lbda)

def tabulate(x, y, f):
    """Return a table of f(x, y)."""
    return np.vectorize(f)(*np.meshgrid(x, y))

delta = 0.1 # param

# Designing a squared grid
x_grid_min = min(min(x_min[0], x_init[0]), min(x_min[1], x_init[1])) - 1.0 # param
y_grid_min = x_grid_min
x_grid_max = max(max(x_min[0], x_init[0]), max(x_min[1], x_init[1])) + 2.0 # param
y_grid_max = x_grid_max

x = np.arange(x_grid_min, x_grid_max, delta)
y = np.arange(y_grid_min, y_grid_max, delta)
X, Y = np.meshgrid(x, y)

Z = tabulate(x, y, loss) # compute loss at each point of the grid


## Plot 2D contour figure
plt.figure()
nb_ellipses = 40 # param
CS = plt.contour(X, Y, Z, nb_ellipses, colors="k")

cm = plt.cm.get_cmap("RdYlBu") # param
sc = plt.scatter(beta0, beta1,
                c=np.arange(beta0.shape[0]),
                marker="o", s=60, cmap=cm, zorder=2) # param
clb = plt.colorbar(sc)
clb.ax.set_title("Iterations")

plt.scatter(beta_star[0], beta_star[1], s=100, linewidth=3, c="limegreen", marker="x", zorder=2) # numerical solution

# plt.title(r"$\gamma = 1 / L$")

save_figure = False
if save_figure:
    plt.savefig("outputs/test.pdf", bbox_inches="tight")
plt.show()