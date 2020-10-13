import os
import pandas as pd

class ConvTable:
    def __init__(self, df):
        self.df = df


if __name__ == "__main__":
    folder = os.path.join(os.getcwd(), "conv_tables")
    full_path = os.path.join(folder, "conv_2D.csv")
    df = pd.read_csv(full_path)

