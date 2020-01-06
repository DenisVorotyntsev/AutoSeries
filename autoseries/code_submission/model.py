import pickle
import os
import time
import copy
from typing import List, Dict, Tuple
from pprint import pprint
import random

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 500)
from sklearn.model_selection import ParameterGrid

from models import LGBModel
from utils import save_float_mem, parse_time, generate_additional_num_features, \
    get_lag_features, get_batch_id, cat_and_batch_id


MAX_SHIFT = 9
TARGET_LAGS = (1, 2, 3, 5, 7)
FEATURES_LAGS = (1, 2, 3)


class Model:
    def __init__(self, info: Dict, test_timestamp: pd.Series, pred_timestamp: pd.Series):
        self.info = info
        self.primary_timestamp = info['primary_timestamp']
        self.primary_id = info['primary_id']
        self.label = info['label']
        self.schema = info['schema']
        self.full_data = None
        self.batch_size = None

        self.best_params = None
        self.training_time_coef = 1.2  # time pillow for updating (more data) & fit after valid (more data)

        self.train_columns = None
        self.columns_dtypes = None

        print(f"\ninfo: {self.info}")

        self.dtype_cols = {}
        self.dtype_cols['cat'] = [col for col, types in self.schema.items() if types == 'str']
        self.dtype_cols['idd'] = self.primary_id
        if len(self.dtype_cols['idd']) > 0:
            self.dtype_cols['cat'] += ["batch_id"]
        self.dtype_cols['num'] = [col for col, types in self.schema.items() if types == 'num']

        self.test_timestamp = test_timestamp
        self.pred_timestamp = pred_timestamp

        self.n_test_timestamp = len(pred_timestamp)
        self.n_retrain = None
        self.update_interval = None
        self.save_time = None
        self.load_time = None

        print(f"sample of test record: {len(test_timestamp)}")
        print(f"number of pred timestamp: {len(pred_timestamp)}")

        self.top_num_features = None
        self.top_cat_features = None

        self.worst_training_time = -1
        self.loss = float("inf")
        self.model = None
        self.models_report = []
        self.time_for_training = None
        self.n_predict = 0

        self.time_pillow_coef = 0.8  # general time pillow
        self.training_time_left = None
        self.updating_time_left = None
        self.predicting_time_left = None
        self.saving_time_left = None
        self.loading_time_left = None

        print(f"Finish init\n")

    def determine_important_features(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            topn: int = 3
    ):
        """
        Determine most important numerical features.
        Model - LGB, importance type - gain.
        :param X: Train data
        :param y: Target
        :param topn: Number of top numerical features to save
        :return:
        """
        params = {
            "id_columns": self.dtype_cols['idd'],
            "cat_columns": self.dtype_cols['cat'],
            "encode_type": "category",
            "target_process": "none",
            "prev_target_col": f"{self.label}__shift-1",
            "apply_weights": "none",
            "use_features": "all"
        }
        model_params = {
            "num_boost_round": 10,
            "learning_rate": 0.05,
            "random_state": 42
        }
        # fit model
        model = LGBModel(**{"params": params, "model_params": model_params})
        _ = model.fit(X, y)

        # select most {topn} important ones
        num_features = [col for col in list(model.feature_importance["feature"]) if col in self.dtype_cols['num']]
        cat_features = [
            col for col in list(model.feature_importance["feature"]) if (col in self.dtype_cols['cat'])
                                                                        & (col != "batch_id")
        ]

        self.top_num_features = list(num_features[:topn])
        self.top_cat_features = list(cat_features[:topn])

        # show results
        print("\n\n" + "-"*60)
        print("Most important num features:")
        print(self.top_num_features)
        print("-"*60 + "\n\n")

    def validate_fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            model_kw: Dict
    ):
        """
        Validate & train model
        :param X: Data
        :param y: Target
        :param model_kw: pipelines parameters dictionary
        :return: Saves best model in self
        """
        print("\n" + "-" * 60)
        model = LGBModel(**model_kw)
        loss, t_val = model.validate(X, y)
        self.training_time_left = self.training_time_left - t_val
        if t_val > self.worst_training_time:
            self.worst_training_time = t_val

        print(f"Time left:                 {self.training_time_left:.2f} s")
        print(f"Time per validation:       {t_val:.2f} s")
        print(f"Worst time:                {self.worst_training_time:.2f} s")
        print(f"Grid RMSE loss:            {loss:.4f}")
        print(f"Best RMSE loss so far:     {self.loss:.4f}")
        pprint(model_kw)

        # save results
        self.models_report.append([loss, model_kw])

        is_better = self.loss > loss
        have_time_to_retrain = self.training_time_left > (t_val*self.training_time_coef)

        # if have too little time - save validation run
        if is_better & (not have_time_to_retrain):
            self.loss = copy.deepcopy(loss)
            self.time_for_training = copy.deepcopy(t_val)
            self.model = copy.deepcopy(model)
            self.best_params = copy.deepcopy(model.get_kw())
            print("New model is better, but we have too little time. Saved val run.")
        # if model is better and have enough - refit on full data
        elif is_better & have_time_to_retrain:
            t_fit = model.fit(X, y)
            self.training_time_left = self.training_time_left - t_fit
            if t_fit > self.worst_training_time:
                self.worst_training_time = t_fit
            print(f"Time per training: {t_fit:.2f}, time left: {self.training_time_left:.2f} s")

            self.loss = copy.deepcopy(loss)
            self.time_for_training = copy.deepcopy(t_fit)
            self.model = copy.deepcopy(model)
            self.best_params = copy.deepcopy(model.get_kw())
            print("New model is better, refitted it.")
        else:
            print("New model is worse, keep old params.")
        print("-" * 60 + "\n")

    def train_baseline(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            model_kw: Dict = None
    ):
        """
        Validate & train baseline model
        :param X: Data
        :param y: Target
        :param model_kw: pipelines parameters dictionary
        :return: Saves best model in self
        """
        if model_kw is None:
            model_kw = dict()
            model_kw["params"] = {
                "id_columns": self.dtype_cols['idd'],
                "cat_columns": self.dtype_cols['cat'],
                "encode_type": "catboost",
                "target_process": "none",
                "prev_target_col": f"{self.label}__shift-1",
                "apply_weights": "none",
                "use_features": "all"
            }
        self.validate_fit(X, y, model_kw)

    def optimize_params(
            self,
            X: pd.DataFrame,
            y: pd.Series
    ):
        """
        Optimize main parameters of pipeline
        :param X: Data
        :param y: Target
        :return: Saves best model in self
        """
        param_grid = ParameterGrid({
            "id_columns": [self.dtype_cols['idd']],
            "cat_columns": [self.dtype_cols['cat']],
            "encode_type": ["catboost"],
            "target_process": ["diff", "none"],
            "prev_target_col": [f"{self.label}__shift-1"],
            "apply_weights": ["none"],
            "use_features": ["all"]
        })
        param_grid = list(param_grid)
        param_grid = [params for params in param_grid if params != self.best_params["params"]]
        random.shuffle(param_grid)
        print(f"Number of grid points: {len(param_grid)}")
        pprint(param_grid)

        # optimize params
        while (self.training_time_left > self.worst_training_time) & (len(param_grid) > 0):
            params = param_grid.pop()
            self.validate_fit(X, y, {"params": params})

    def select_features(
            self,
            X: pd.DataFrame,
            y: pd.Series
    ):
        """
        Select features (importance-based)
        :param X: Data
        :param y: Target
        :return: Saves best model in self
        """
        sorted_features = list(self.model.feature_importance["feature"])

        # remove least important features
        bot_n = [0.2, 0.5, 0.75, 0.05, 0.1]
        while (self.training_time_left > self.time_for_training) & (len(bot_n) > 0):
            n_bot = np.ceil(bot_n.pop() * X.shape[1])
            n_bot = int(n_bot)
            params = copy.deepcopy(self.best_params)["params"]
            params["use_features"] = sorted_features[:n_bot]
            self.validate_fit(X, y, {"params": params})

    def optimize_model_params(
            self,
            X: pd.DataFrame,
            y: pd.Series
    ) -> None:
        """
        Optimize params of LightGBM regressor
        :param X: Data
        :param y: Target
        :return: Saves best model in self
        """
        param_grid = ParameterGrid({
            "learning_rate": [0.05],
            "n_estimators": [1000],
            "num_leaves": [15, 31, 63, 127, 255],
            "min_child_samples": [3, 20, 50, 150],

            "subsample_freq": [1, 5, 25, 50],
            "colsample_bytree": [1.0, 0.8, 0.6],
            "subsample": [1.0, 0.8, 0.6],

            "lambda_l2": [0, 0.1, 1, 10],
            "random_state": [2020]
        })
        param_grid = list(param_grid)
        random.shuffle(param_grid)
        print(f"Number of grid points: {len(param_grid)}")

        # optimize params
        while (self.training_time_left > self.worst_training_time) & (len(param_grid) > 0):
            params = param_grid.pop()
            self.validate_fit(X, y, {"model_params": params, "params": self.best_params["params"]})

    def show_training_results(self):
        """
        Show training results
        :return: None
        """
        # show final results
        print("\n\n" + "-"*60 + "\nAll results")
        loses = [report[0] for report in self.models_report]
        ind_sorted = np.argsort(loses)
        for ind in ind_sorted:
            pprint(self.models_report[ind])
        print("\n\n" + "-"*60 + "\n\n" + "Best pipeline")
        print(f"Time for training: {self.time_for_training:.2f}, best loss: {self.loss:.2f}")
        print("Best model:")
        pprint(self.best_params)
        print("\n\n" + "-"*60 + "\n\n")

    def train(self, train_data: pd.DataFrame, time_info):
        t1 = time.time()
        print(f"\nTrain time budget: {time_info['train']}s")
        print("Train data shape:", train_data.shape, "\n\n")

        X = train_data.copy()

        # save mem
        X = save_float_mem(X)

        # batch id
        X = get_batch_id(X, self.dtype_cols['idd'])

        # most important features
        if self.top_num_features is None:
            self.determine_important_features(
                X.drop([self.label]+[self.primary_timestamp], axis=1),
                X[self.label],
                topn=3
            )

        # num features
        X = generate_additional_num_features(X, self.top_num_features)

        # add lag features
        if len(self.dtype_cols['idd']) != 0:
            self.batch_size = int(X[self.dtype_cols['idd']].drop_duplicates().shape[0])
            self.full_data = train_data.tail(self.batch_size * MAX_SHIFT)
        else:
            self.batch_size = 1
            self.full_data = train_data.tail(self.batch_size * MAX_SHIFT)
        print(f"Batch size: {self.batch_size}\n\n")
        X = get_lag_features(X, [self.label], lags=TARGET_LAGS)
        X = get_lag_features(X, self.top_num_features, lags=FEATURES_LAGS)
        # X = get_lag_features(X, self.top_cat_features, lags=(1,), operations=("shift",))

        # cat columns
        self.dtype_cols['cat'] = [
            col for col in X.columns if (col in self.dtype_cols['cat']) or (X[col].dtype == np.dtype('object'))
        ]

        # process cat columns
        X = cat_and_batch_id(X, self.dtype_cols['cat'])

        # parse time features
        time_fea = parse_time(X[self.primary_timestamp])
        X = X.drop(self.primary_timestamp, axis=1)
        X = pd.concat([X, time_fea], axis=1)

        # drop label
        X = X.reset_index(drop=True)
        y = X[self.label].copy()
        X = X.drop(self.label, axis=1)

        # save columns order and dtypes
        self.train_columns = list(X.columns)
        self.columns_dtypes = X.dtypes

        # train model
        t2 = time.time()
        print(f"Processing time: {t2 - t1:.2f}")
        if self.best_params is None:
            # show training data
            print(X.sample(10))
            for col in X.columns:
                print(col, len(set(X[col].dropna())), np.mean(X[col].isnull()))

            self.training_time_left = self.info["time_budget"]["train"] * self.time_pillow_coef - (t2 - t1)
            self.train_baseline(X, y)
            self.optimize_params(X, y)
            self.select_features(X, y)
            self.optimize_model_params(X, y)
            self.show_training_results()
        else:
            print("\n\n" + "-" * 60)
            print("Refit model")
            pprint(self.best_params)
            print("-" * 60 + "\n\n")
            model = LGBModel(**self.best_params)
            model.fit(X, y)
            self.model = model

        # determine number of retraining rounds
        if self.update_interval is None:
            print(f"Time for training: {self.time_for_training} s\n\n")
            n_retrain = np.ceil(self.info["time_budget"]["update"] / self.time_for_training / self.training_time_coef)
            if self.n_test_timestamp <= 1:
                n_retrain = 0
            else:
                n_retrain = np.clip(0, self.n_test_timestamp-1, n_retrain)
            self.n_retrain = int(n_retrain)
            if self.n_retrain > 0:
                self.update_interval = np.ceil(self.n_test_timestamp / self.n_retrain)
                self.update_interval = int(self.update_interval)
            else:
                self.update_interval = np.inf
            print(f"Num retraining: {n_retrain}\n\n")
        return 'predict'

    def predict(self, new_history: pd.DataFrame, pred_record: pd.DataFrame, time_info):
        if self.predicting_time_left is None:
            self.predicting_time_left = self.info["time_budget"]["predict"]
        t1 = time.time()
        self.n_predict += 1

        # save mem
        pred_record = save_float_mem(pred_record)

        # additional num features
        pred_record = generate_additional_num_features(pred_record, self.top_num_features)

        self.full_data = self.full_data.append(new_history, ignore_index=True)
        self.full_data = self.full_data.reset_index(drop=True)
        self.full_data = self.full_data.tail(self.batch_size * MAX_SHIFT)
        print("Full data shape:", self.full_data.shape)
        print("pred record shape:", pred_record.shape)
        pred_record[self.label] = None
        X = pd.concat(
            [self.full_data, pred_record],
            axis=0, ignore_index=True, sort=False
        ).reset_index(drop=True)

        # batch id
        X = get_batch_id(X, self.dtype_cols['idd'])

        # process cat columns
        X = cat_and_batch_id(X, self.dtype_cols['cat'])

        # add lag features
        X = get_lag_features(X, [self.label], lags=TARGET_LAGS)
        X = get_lag_features(X, self.top_num_features, lags=FEATURES_LAGS)
        # X = get_lag_features(X, self.top_cat_features, lags=(1,), operations=("shift",))

        # drop label and get test data with lag features
        X = X.drop(self.label, axis=1)
        pred_record = X.tail(pred_record.shape[0]).reset_index(drop=True)

        # parse time features
        time_fea = parse_time(pred_record[self.primary_timestamp])
        pred_record = pd.concat([pred_record, time_fea], axis=1)
        pred_record = pred_record.drop(self.primary_timestamp, axis=1)

        # check columns order
        pred_record = pred_record[self.train_columns]

        # check dtypes
        for col in self.columns_dtypes.index:
            dtype = self.columns_dtypes[col]
            pred_record[col] = pred_record[col].astype(dtype)

        # make predictions
        predictions = self.model.predict(pred_record)

        if self.n_predict > self.update_interval:
            next_step = 'update'
            self.n_predict = 0
        else:
            next_step = 'predict'
        t2 = time.time()
        time_per_pred = t2 - t1
        self.predicting_time_left = self.predicting_time_left - time_per_pred
        print("\n" + "-" * 60)
        print(f"New time for pred: {time_per_pred:.2f}s")
        print(f"Pred time left: {self.predicting_time_left:.2f}s")
        print(f"Best val RMSE loss: {self.loss:.4f}")
        print("-" * 60 + "\n")
        return list(predictions), next_step

    def update(self, train_data: pd.DataFrame, test_history_data: pd.DataFrame, time_info):
        if self.updating_time_left is None:
            self.updating_time_left = self.info["time_budget"]['update'] * self.time_pillow_coef

        print(f"\nUpdate time left: {self.updating_time_left:.2f} s\n\n")
        if self.updating_time_left > (self.time_for_training * self.training_time_coef):
            t1 = time.time()
            total_data = pd.concat(
                [train_data, test_history_data],
                axis=0, sort=False, ignore_index=True
            ).reset_index(drop=True)
            self.train(total_data, time_info)
            t2 = time.time()
            self.time_for_training = t2 - t1
            self.updating_time_left = self.updating_time_left - self.time_for_training
            print("\n" + "-"*60)
            print("Finish update")
            print(f"New time for training: {self.time_for_training:.2f}s")
            print(f"Update time left: {self.updating_time_left:.2f}s")
            print("-"*60 + "\n")
        else:
            print("Not enough time for updating!")
        next_step = 'predict'
        return next_step

    def save(self, model_dir: str, time_info):
        if self.saving_time_left is None:
            self.loading_time_left = self.info["time_budget"]["load"] * self.time_pillow_coef
            self.saving_time_left = self.info["time_budget"]["save"] * self.time_pillow_coef

        t1 = time.time()
        pkl_list = []
        for attr in dir(self):
            if attr.startswith('__') or attr in ['train', 'predict', 'update', 'save', 'load']:
                continue
            pkl_list.append(attr)
            pickle.dump(getattr(self, attr), open(os.path.join(model_dir, f'{attr}.pkl'), 'wb'))
        pickle.dump(pkl_list, open(os.path.join(model_dir, f'pkl_list.pkl'), 'wb'))
        t2 = time.time()
        dt = t2 - t1

        self.save_time = dt
        self.saving_time_left = self.saving_time_left - self.save_time

        print("\n" + "-" * 60)
        print("Finish saving")
        print(f"New time for saving: {self.save_time:.2f}s")
        print(f"Saving time left: {self.saving_time_left:.2f}s")
        print("-" * 60 + "\n")

        if self.save_time > self.saving_time_left:
            self.update_interval = np.inf
            print("No more time for updating (saving)")

    def load(self, model_dir: str, time_info):
        t1 = time.time()
        pkl_list = pickle.load(open(os.path.join(model_dir, 'pkl_list.pkl'), 'rb'))
        for attr in pkl_list:
            setattr(self, attr, pickle.load(open(os.path.join(model_dir, f'{attr}.pkl'), 'rb')))
        t2 = time.time()
        dt = t2 - t1

        self.load_time = dt
        print(self.loading_time_left, self.load_time, self.update_interval)
        self.loading_time_left = self.loading_time_left - self.load_time

        print("\n" + "-" * 60)
        print("Finish loading")
        print(f"New time for loading: {self.load_time}s")
        print(f"Loading time left: {self.loading_time_left}s")
        print("-" * 60 + "\n")

        if self.load_time > self.loading_time_left:
            self.update_interval = np.inf
            print("No more time for updating (loading)")
