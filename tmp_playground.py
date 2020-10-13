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



