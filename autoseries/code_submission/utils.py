from typing import List, Tuple

import pandas as pd
import numpy as np


def cat_and_batch_id(
        X: pd.DataFrame,
        cat_cols: List[str]
) -> pd.DataFrame:
    """
    Process categories in data by adding batch id to each category
    :param X: Train/test dataframe
    :param cat_cols: categorical columns in dataframe
    :return: Dataframe with processed categories
    """
    if "batch_id" in X.columns:
        for col in cat_cols:
            X[col] = X[col].astype("str") + "__batchid:" + X["batch_id"].astype("str")
    return X


def save_float_mem(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Save memory by changing "float64" (standard format) to "float32"
    :param df: Data
    :return: Processed data
    """
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype == "float":
            df[col] = df[col].astype("float32")
    return df


def parse_time(
        xtime: pd.Series
) -> pd.DataFrame:
    """
    Create time-based features
    :param xtime: UNIX time
    :return: Dataframe of time-based features
    """
    result = pd.DataFrame()
    dtcol = pd.to_datetime(xtime, unit='s')

    result[f'{xtime.name}_year'] = dtcol.dt.year
    result[f'{xtime.name}_month'] = dtcol.dt.month
    result[f'{xtime.name}_dayofyear'] = dtcol.dt.dayofyear
    result[f'{xtime.name}_weekday'] = dtcol.dt.weekday
    result[f'{xtime.name}_hour'] = dtcol.dt.hour

    try:
        result[f'{xtime.name}_year'] = result[f'{xtime.name}_year'].astype("int16")
        result[f'{xtime.name}_month'] = result[f'{xtime.name}_month'].astype("int8")
        result[f'{xtime.name}_dayofyear'] = result[f'{xtime.name}_dayofyear'].astype("int16")
        result[f'{xtime.name}_weekday'] = result[f'{xtime.name}_weekday'].astype("int8")
        result[f'{xtime.name}_hour'] = result[f'{xtime.name}_hour'].astype("int8")
    except:
        # None in results
        pass
    return result


def generate_additional_num_features(
        X: pd.DataFrame,
        num_features: List[str]
) -> pd.DataFrame:
    """
    Generate additional numerical features by applying mathematical operations on combinations of features
    :param X: Data
    :param num_features: Numerical features in data to use for combinations
    :return: Dataframe with new numerical features
    """
    if len(num_features) == 0:
        return X

    new_cols = []
    for i in range(len(num_features) - 1):
        for j in range(1, len(num_features)):
            col1, col2 = num_features[i], num_features[j]
            X[f"{col1}_add_{col2}"] = X[col1] + X[col2]
            X[f"{col1}_div_{col2}"] = X[col1] / X[col2]
            X[f"{col1}_mul_{col2}"] = X[col1] * X[col2]
            X[f"{col1}_min_{col2}"] = X[col1] - X[col2]

    for col in new_cols:
        X[col] = X[col].astype("float32")
    return X


def get_lag_features(
        X: pd.DataFrame,
        cols: List[str],
        lags: Tuple[int] = (1, 2, 3, 7, 14),
        operations: Tuple[str] = ("diff", "shift"),
) -> pd.DataFrame:
    """
    Get lag and diff features (for time-series nature of data)
    :param X: Data
    :param cols: Features for diff/shift
    :param lags: Steps for shift/diff
    :param operations: Operations to perform, should be "diff" or "shift" or both
    :return: Dataframe with new lag/shift features
    """
    for col in cols:
        if "batch_id" in X.columns:
            df = X.groupby("batch_id")
        else:
            df = X

        for lag in lags:
            if "shift" in operations:
                X[f"{col}__shift-{lag}"] = df[col].shift(lag).values

            if "diff" in operations:
                values = df[col].shift(1).values
                X[f"{col}__diff-{lag}"] = values - df[col].shift(lag+1).values
    return X


def get_batch_id(
        X: pd.DataFrame,
        idd_cols: List[str]
) -> pd.DataFrame:
    """
    Add "batch id" columsn to dataframe
    :param X: Data
    :param idd_cols: Id columns in data
    :return: Dataframe with "batch_id" column, if there is at least one id col in dataframe.
    If not - return initial dataframe.
    """
    if len(idd_cols) != 0:
        X["batch_id"] = ""
        for col in idd_cols:
            X["batch_id"] = X["batch_id"] + "__" + X[col].astype("str")
    return X
