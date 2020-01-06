from typing import List, Tuple, Dict
import time
from pprint import pprint

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from catboost_enc import CatBoostEncoder
from encoders import CategoryLabels, TimeSeriesCatboostEncoder


class LGBModel:
    def __init__(
            self,
            params: Dict = None,
            model_params: Dict = None
    ) -> None:
        self.params = params
        if model_params is None:
            self.model_params = {
                "num_boost_round": 1000,
                "learning_rate": 0.05,

                "subsample_freq": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,

                "random_state": 2020
            }
        else:
            self.model_params = model_params

        self.process_target_params = {}
        self.cat_cols_processor = None

        self.model = None
        self.counter = 0
        self.val_loss = None

        self.feature_importance = None

    def get_kw(
            self
    ) -> Dict:
        """Return training params of Model"""
        return {"params": self.params, "model_params": self.model_params}

    @staticmethod
    def calc_loss(
            y_true: np.ndarray,
            y_hat: np.ndarray
    ) -> float:
        """
        Calculate RMSE loss
        :param y_true: True labels
        :param y_hat: Predictions
        :return: RMSE loss
        """
        loss = mean_squared_error(y_true, y_hat) ** 0.5
        return loss

    def process_categories(
            self,
            X: pd.DataFrame,
            y: np.ndarray = None,
            is_train:
            bool = True
    ) -> pd.DataFrame:
        """
        Process categories. Possible processing:
            1. Encode as category (.astype("category"))
            2. Catboost encoding (non shuffled)
        :param X: data
        :param y: target
        :param is_train: True if train data, False if val or test data
        :return: Processed dataframe
        """
        if is_train:
            if self.params["encode_type"] == "catboost":
                self.cat_cols_processor = CatBoostEncoder(cols=self.params["cat_columns"])
            elif self.params["encode_type"] == "category":
                self.cat_cols_processor = CategoryLabels(cols=self.params["cat_columns"])
            else:
                raise NotImplementedError
            X = self.cat_cols_processor.fit_transform(X, y)
        else:
            X = self.cat_cols_processor.transform(X)
        return X

    def process_target(
            self, y: np.ndarray,
            X: pd.DataFrame = None,
            stage: str = "pre"
    ) -> np.ndarray:
        """
        Pre and post-process target
        :param y: target
        :param X: Train or test dataframe
        :param stage: "pre" for pre-processing of "post" for post-processing
        :return: Processed target
        """
        y_processed = y.copy()
        if self.params["target_process"] == "**0.5":
            if stage == "pre":
                y_processed = y_processed ** 0.5
            if stage == "post":
                y_processed = y_processed ** 2
        elif self.params["target_process"] == "diff":
            prev_target_col = self.params["prev_target_col"]
            if stage == "pre":
                y_processed = y_processed - X[prev_target_col]
                y_processed.loc[X[prev_target_col].isnull()] = 0
            if stage == "post":
                print(f"Fraction of missing values in X: {np.mean(X[prev_target_col].isnull())}")
                y_processed = y_processed + X[prev_target_col]
                y_processed = y_processed.fillna(self.process_target_params["mean_target"])
        elif self.params["target_process"] == "none":
            y_processed = y.copy()
        else:
            raise NotImplementedError
        return y_processed

    def get_train_weights(
            self,
            X: pd.DataFrame
    ) -> np.array:
        """
        Prepare samples weights
        :param X: Train data
        :return: Array of samples weights, same length as given data
        """
        if self.params["apply_weights"] == "none":
            train_w = np.ones(X.shape[0])
        elif self.params["apply_weights"] == "linear":
            train_w = np.linspace(0, 1, num=X.shape[0])
        else:
            raise NotImplementedError
        return train_w

    def get_train_features(
            self,
            X: pd.DataFrame
    ) -> List[str]:
        """
        Select features to use in training, which is an intersect of given features and features in given dataframe.
        :param X: Train data
        :return: List of features to use in training
        """
        if self.params["use_features"] == "all":
            features_to_use = list(X.columns)
        else:
            features_to_use = [col for col in X.columns if col in self.params["use_features"]]
        return features_to_use

    def get_model_importance(
            self
    ) -> None:
        """
        Get features importance of fitted model, sort features by importance (high -> low).
        :return: Save in self features importance dataframe.
        """
        importance_df = pd.DataFrame({})
        importance_df["feature"] = self.model.booster_.feature_name()
        importance_df["importance"] = self.model.booster_.feature_importance(importance_type="gain")
        importance_df = importance_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
        self.feature_importance = importance_df

    def validate(
            self,
            X: pd.DataFrame,
            y: pd.Series
    ) -> Tuple[float, float]:
        """
        Validate model and save fitted model
        :param X: Full train data, which will be split into train/val by time
        :param y: Full train labels
        :return: RMSE loss of pipeline and spent time
        """
        t1 = time.time()

        # shift target by a constant (min value)
        # allow to use various target processing (which work only with positive nums)
        self.process_target_params["target_shift"] = np.min(y) - 1
        y = y - self.process_target_params["target_shift"]

        # split into train and val
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.1, shuffle=False)
        X_train, X_eval = X_train.reset_index(drop=True), X_eval.reset_index(drop=True)
        y_train, y_eval = y_train.reset_index(drop=True), y_eval.reset_index(drop=True)

        # process target
        self.process_target_params["mean_target"] = np.mean(y)
        y_train = self.process_target(y_train, X_train, stage="pre")
        y_eval = self.process_target(y_eval, X_eval, stage="pre")

        # process categories
        X_train = self.process_categories(X_train, y_train, is_train=True)
        X_eval = self.process_categories(X_eval, is_train=False)

        # train sample weights
        train_w = self.get_train_weights(X_train)

        # get train features (most important ones)
        features_to_use = self.get_train_features(X_train)

        # train model
        self.model = lgb.LGBMRegressor(**self.model_params)
        self.model.fit(
            X_train[features_to_use], y_train, sample_weight=train_w,
            eval_set=(X_eval[features_to_use], y_eval),
            early_stopping_rounds=50, verbose=None
        )
        self.model_params["num_boost_round"] = self.model.best_iteration_
        y_hat = self.model.predict(X_eval[features_to_use])

        # post-process target
        y_hat = self.process_target(y_hat, X_eval, stage="post")
        y_eval = self.process_target(y_eval, X_eval, stage="post")

        # remove shift
        y_hat = y_hat + self.process_target_params["target_shift"]
        y_eval = y_eval + self.process_target_params["target_shift"]

        # calc loss
        loss = self.calc_loss(y_eval, y_hat)

        # feature importance
        self.get_model_importance()

        t2 = time.time()
        dt = t2 - t1
        return loss, dt

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series
    ) -> float:
        """
        Fit model
        :param X: Train data
        :param y: Train labels
        :return: Spent time on training
        """
        t1 = time.time()
        # shift target by a constant (min value)
        # allow to use various target processing (which work only with positive nums)
        self.process_target_params["target_shift"] = np.min(y) - 1
        y = y - self.process_target_params["target_shift"]

        # pre-process target
        self.process_target_params["mean_target"] = np.mean(y)
        y = self.process_target(y, X, stage="pre")

        # process categories
        X = self.process_categories(X, y, is_train=True)

        # train sample weights
        train_w = self.get_train_weights(X)

        # get train features
        features_to_use = self.get_train_features(X)

        # fit model with opt number of trees
        self.model = lgb.LGBMRegressor(**self.model_params)
        self.model.fit(X[features_to_use], y, sample_weight=train_w)

        # feature importance
        self.get_model_importance()

        t2 = time.time()
        dt = t2 - t1
        return dt

    def predict(
            self,
            X: pd.DataFrame
    ) -> np.array:
        """
        Make predictions
        :param X: test data
        :return: predictions array
        """
        # process categories
        X = self.process_categories(X, is_train=False)

        # get train features
        features_to_use = self.get_train_features(X)

        # predict
        y_hat = np.array(self.model.predict(X[features_to_use]))

        # post-process target
        y_hat = self.process_target(y_hat, X, stage="post")

        # remove shift
        y_hat = y_hat + self.process_target_params["target_shift"]

        self.counter += 1
        if self.counter == 10:
            # show test data, just in case
            print("\n\n"+"-"*60)
            print("X columns:")
            pprint(X.columns)
            print(X.tail())
            print("Predictions: ")
            print(y_hat[-10:])
            print("-" * 60+"\n\n")
        return y_hat
