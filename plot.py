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

def compute_2d_solution(f, grad_f):
    try: # computing numerical solution
        beta_star, _, _ = fmin_l_bfgs_b(f, np.zeros(2), grad_f)
        print("Solution computed with L-BFGS-B.")
        return beta_star
    except: # unknown solution
        print("Impossible to compute the solution.")

def set_2d_grid(beta0, beta1, beta_star, delta=.1):
    """
    Create two arrays representing a 2D mesh.

    Parameters
    ----------
    beta0 : ndarray
        Array of x-axis values of the iterates.
    beta1 : ndarray
        Array of y-axis values of the iterates.
    beta_star : ndarray of shape (2,) or None
        Coordinates of the solution.
    delta : float, default=0.1
        Size of the mesh. Same for x- and y-axis.

    Returns
    ----------
    x : ndarray
        array of x-axis of the mesh.
    y : ndarray
        array of y-axis of the mesh.
    """
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

    x = np.arange(x_grid_min, x_grid_max, delta)
    y = np.arange(y_grid_min, y_grid_max, delta)
    return x, y

def plot_2d_color_bar(sc, progress_measure, color_seq):
    cb = plt.colorbar(sc) #, format="%d")
    cb.set_ticks(color_seq)
    cb.set_ticklabels(color_seq)
    cb.ax.set_title(progress_measure.capitalize())


def tabulate_2d(x, y, f):
    """Returns a table of f([x, y]) or f(x, y)."""
    beta0, beta1 = 0., 0.
    try:
        f(beta0, beta1)
        # g = f
    except TypeError:
        # Function of array of shape (2,) to function of two floats
        # print("Function of array of shape (2,) to function of two floats.")
        g = lambda beta0, beta1: f(np.array([beta0, beta1]))
    else:
        print("Function f passed requires incorrect inputs.")
    return np.vectorize(g)(*np.meshgrid(x, y))


def plot_2d(df, f, beta_star=None, progress_measure="iter", grad_f=None):
    beta0 = df["beta0"].values
    beta1 = df["beta1"].values

    # Try to compute solution with L-BFGS-B if not given
    if beta_star is None:
        beta_star = compute_2d_solution(f, grad_f)

    # Designing a squared grid
    x, y = set_2d_grid(beta0, beta1, beta_star)
    X, Y = np.meshgrid(x, y)

    # Compute loss at each point of the grid
    Z = tabulate_2d(x, y, f)
    print(Z.shape)

    # Plot 2D contour figure
    fig, ax = plt.subplots(figsize=(5, 3))
    nb_ellipses = 20 # param
    CS = ax.contour(X, Y, Z, nb_ellipses, colors="k")

    cm = plt.cm.get_cmap("RdYlBu") # param

    if progress_measure in ["iteration", "epoch", "time"]:
        color_seq = df[progress_measure].values
    else:
        raise ValueError(f"Progress measure {progress_measure} not supported. Must be iteration, epoch or time.")

    # TODO: Scatter only every ~10/20% of the iterations / Plot only 30 points
    sc = ax.scatter(beta0, beta1, c=color_seq, marker="o", s=60, cmap=cm, zorder=2) # param
    # ax.set_title(r"$\gamma = 1 / L$") # Step size gradient descent

    plot_2d_color_bar(sc, progress_measure, color_seq)

    if beta_star is not None:
        ax.scatter(beta_star[0], beta_star[1], s=70, linewidth=2, c="blue", marker="x", zorder=2)

    fig.tight_layout()

    return fig, ax


@click.command()
@click.argument("progress_measure")
@click.option("--save/--no-save", default=True, help="Save the example 2D plot.")
# @click.option("--save/--no-save", default=True, help="Save the example 2D plot.")
def example_2d_plot(progress_measure, save):
    folder = os.path.join(os.getcwd(), "conv_tables")
    full_path = os.path.join(folder, "conv_2D.csv")
    df = pd.read_csv(full_path)

    # Computing numerical solution
    beta_star, _, _ = fmin_l_bfgs_b(f, np.zeros(2), grad_f)

    # print("No solution, no gradient passed")
    # fig, ax = plot_2d(df, f, progress_measure=progress_measure)
    print("Gradient passed")
    fig, ax = plot_2d(df, f, progress_measure=progress_measure, grad_f=grad_f)
    # print("Solution passed")
    # fig, ax = plot_2d(df, f, progress_measure=progress_measure, beta_star=beta_star)
    # print("Solution and gradient passed")
    # fig, ax = plot_2d(df, f, progress_measure=progress_measure, beta_star=beta_star, grad_f=grad_f)

    if save:
        file_name = "example_plot_2d_" + progress_measure + ".pdf"
        plt.savefig(os.path.join("outputs/", file_name), bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    example_2d_plot()
