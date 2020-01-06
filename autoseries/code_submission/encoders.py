from typing import List

import numpy as np
import pandas as pd


class CategoryLabels:
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit_transform(self, X: pd.DataFrame, y:pd.Series = None):
        for col in self.cols:
            X[col] = X[col].astype("str")
            X[col] = X[col].astype("category")
        return X

    def transform(self, X: pd.DataFrame):
        for col in self.cols:
            X[col] = X[col].astype("str")
            X[col] = X[col].astype("category")
        return X


class TimeSeriesCatboostEncoder:
    def __init__(self, cols):
        self.cols = cols

        self.target_name = "__TARGET__"
        self.mean_target = None
        self.mapping = {}

    def fit_transform(self, X, y):
        X_copy = X[self.cols].copy()
        X_copy[self.target_name] = y

        self.mean_target = np.mean(y)
        for col in self.cols:
            grouped = X_copy.groupby(col)[self.target_name]

            # test
            self.mapping[col] = grouped.mean().to_dict()

            # train
            counts, sums = grouped.cumcount(), grouped.cumsum()
            X[col] = (sums - X_copy[self.target_name]) / counts
            X.loc[counts == 0, col] = None
        return X

    def transform(self, X):
        for col in self.cols:
            X[col] = X[col].map(self.mapping[col])
            X[col] = X[col].fillna(self.mean_target)
        return X
