"""
  AutoWSL datasets.
"""
import copy
from os.path import join
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
from common import get_logger

TYPE_MAP = {
    'cat': str,
    'multi-cat': str,
    'str': str,
    'num': np.float64,
    'timestamp': 'str'
}

VERBOSITY_LEVEL = 'WARNING'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)
PRIMARY_TIMESTAMP = 'primary_timestamp'
LABEL_NAME = 'label'
TIMESTAMP_TYPE_NAME = 'timestamp'
TRAIN_FILE = 'train.data'
TEST_FILE = 'test.data'
TIME_FILE = 'test_time.data'
INFO_FILE = 'info.yaml'


def _date_parser(millisecs):
    if np.isnan(float(millisecs)):
        return millisecs

    return datetime.fromtimestamp(float(millisecs))


class Dataset:
    """"Dataset"""
    def __init__(self, dataset_dir):
        """
            train_dataset, test_dataset: list of strings
            train_label: np.array
        """
        self.dataset_dir_ = dataset_dir
        self.metadata_ = self._read_metadata(join(dataset_dir, INFO_FILE))
        self.train_dataset = None
        self.test_dataset = None

        self.get_train()
        self.get_test()
        self._pred_time = self._get_pred_time()
        self._primary_timestamp = self.metadata_[PRIMARY_TIMESTAMP]

    def get_train(self):
        """get train"""
        if self.train_dataset is None:
            self.train_dataset = self._read_dataset(
                join(self.dataset_dir_, TRAIN_FILE))
        return copy.deepcopy(self.train_dataset)

    def get_test(self):
        """get test"""
        if self.test_dataset is None:
            self.test_dataset = self._read_dataset(
                join(self.dataset_dir_, TEST_FILE))
        return copy.deepcopy(self.test_dataset)

    def get_metadata(self):
        """get metadata"""
        return copy.deepcopy(self.metadata_)

    def is_end(self, idx):
        """whether time idx is the end of data"""
        return idx == len(self._pred_time)

    def _get_period(self, idx1, idx2):
        next_time = self._get_time_point(idx2)
        timestamp = self.test_dataset[self._primary_timestamp]
        select = timestamp < next_time
        if idx1 is not None:
            last_time = self._get_time_point(idx1)
            select &= timestamp >= last_time
        return self.test_dataset[select]

    def get_history(self, idx):
        """get the new history before time idx"""
        if idx > 0:
            ret = self._get_period(idx-1, idx)
        else:
            ret = self._get_period(None, idx)
        return copy.deepcopy(ret)

    def get_next_pred(self, idx):
        """get the next pred time point (idx) (maybe batch data)"""
        next_time = self._get_time_point(idx)
        select = self.test_dataset[self._primary_timestamp] == next_time
        data = self.test_dataset[select].drop(
            self.metadata_[LABEL_NAME], axis=1)
        return copy.deepcopy(data)

    def get_all_history(self, idx):
        """get all history before idx"""
        return copy.deepcopy(self._get_period(None, idx))

    def _get_pred_time(self):
        """get the pred time point"""
        return pd.read_csv(join(self.dataset_dir_, TIME_FILE),
                           parse_dates=[self.metadata_[PRIMARY_TIMESTAMP]],
                           date_parser=_date_parser)

    def _get_time_point(self, idx):
        return self._pred_time.iloc[idx, 0]

    @staticmethod
    def _read_metadata(metadata_path):
        with open(metadata_path, 'r') as ftmp:
            return yaml.safe_load(ftmp)

    def _read_dataset(self, dataset_path):
        schema = self.metadata_['schema']
        table_dtype = {key: TYPE_MAP[val] for key, val in schema.items()}
        date_list = [key for key, val in schema.items()
                     if val == TIMESTAMP_TYPE_NAME]
        dataset = pd.read_csv(
            dataset_path, sep='\t', dtype=table_dtype,
            parse_dates=date_list, date_parser=_date_parser)

        return dataset

    def get_train_num(self):
        """ return the number of train instance """
        return self.metadata_["train_num"]

    def get_test_num(self):
        """ return the number of test instance """
        return self.metadata_["test_num"]

    def get_test_timestamp(self):
        """get timestamps of test data"""
        return copy.deepcopy(self.test_dataset[self._primary_timestamp])

    def get_pred_timestamp(self):
        """get timestamps of pred data"""
        return copy.deepcopy(self._pred_time)
