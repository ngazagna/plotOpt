#!/usr/bin/python

import os
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loss import f, grad_f
from scipy.optimize import fmin_l_bfgs_b

########################################################################
def plot_epochs(monitors, solvers):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for monit in monitors:
        plt.semilogy(monit.obj, lw=2)
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("objective")

    plt.legend(solvers)

    plt.subplot(1, 2, 2)

    for monit in monitors:
        plt.semilogy(monit.err, lw=2)
        plt.title("Distance to optimum")
        plt.xlabel("Epoch")
        plt.ylabel("$\|x_k - x^*\|_2$")

    plt.legend(solvers)

def plot_time(monitors, solvers):
    for monit in monitors:
        objs = monit.obj
        plt.semilogy(np.linspace(0, monit.total_time, len(objs)), objs, lw=2)
        plt.title("Loss")
        plt.xlabel("Timing")
        plt.ylabel("$f(x_k) - f(x^*)$")

    plt.legend(solvers)


def plot_iteration(monitors, solvers):
    for monit in monitors:
        objs = monit.obj
        plt.semilogy(np.linspace(0, monit.iteration, len(objs)), objs, lw=2)
        plt.title("Loss")
        plt.xlabel("Iteration")
        plt.ylabel("$f(x_k) - f(x^*)$")

    plt.legend(solvers)
########################################################################

def tabulate(x, y, f):
    """Return a table of f(x, y)."""
    return np.vectorize(f)(*np.meshgrid(x, y))

def plot_2d(df, f, beta_star=None):
    beta0 = df["beta0"].values
    beta1 = df["beta1"].values

    # Designing a squared grid
    if beta_star is None:
        x_grid_min = min(beta0) - 1.
        x_grid_max = max(beta0) + 1.
        y_grid_min = min(beta1) - 1.
        y_grid_max = max(beta1) + 1.
    else:
        x_grid_min = min(beta_star[0], min(beta0)) - 1.
        x_grid_max = max(beta_star[0], max(beta0)) + 1.
        y_grid_min = min(beta_star[1], min(beta1)) - 1.
        y_grid_max = max(beta_star[1], max(beta1)) + 1.

    delta = 0.1 # param
    x = np.arange(x_grid_min, x_grid_max, delta)
    y = np.arange(y_grid_min, y_grid_max, delta)
    X, Y = np.meshgrid(x, y)

    # Compute loss at each point of the grid
    # Z = f([x, y]) # def f(beta):
    f_2_args = lambda beta0, beta1: f(np.array([beta0, beta1]))
    Z = tabulate(x, y, f_2_args)
    # Z = tabulate(x, y, f) # def f(x, y):

    ## Plot 2D contour figure
    fig, ax = plt.subplots(figsize=(5, 3))
    nb_ellipses = 40 # param
    CS = ax.contour(X, Y, Z, nb_ellipses, colors="k")

    cm = plt.cm.get_cmap("RdYlBu") # param
    sc = ax.scatter(beta0, beta1, c=np.arange(beta0.shape[0]), marker="o", s=60, cmap=cm, zorder=2) # param
    cb = plt.colorbar(sc, format="%d")
    cb.ax.set_title("Iterations")

    if beta_star is not None:
        ax.scatter(beta_star[0], beta_star[1], s=100, linewidth=3, c="limegreen", marker="x", zorder=2) # numerical solution
    fig.tight_layout()

    return fig, ax


@click.command()
@click.option("--save/--no-save", default=True, help="Save the example 2D plot.")
def example_2d_plot(save):
    folder = os.path.join(os.getcwd(), "conv_tables")
    full_path = os.path.join(folder, "conv_2D.csv")
    df = pd.read_csv(full_path)

    # Computing numerical solution
    beta_star, f_star, _ = fmin_l_bfgs_b(f, np.zeros(2), grad_f)

    fig, ax = plot_2d(df, f, beta_star)
    # plt.title(r"$\gamma = 1 / L$")

    # save = False
    if save:
        plt.savefig("outputs/example_plot_2d.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    example_2d_plot()
