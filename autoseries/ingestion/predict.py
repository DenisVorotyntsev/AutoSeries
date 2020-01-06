# pylint: disable=logging-fstring-interpolation, broad-except
"""prediction"""
import argparse
import os
from os.path import join
from sys import path
import pandas as pd
import yaml

from common import get_logger, init_usermodel
from timing import Timer, TimeoutException
from dataset import Dataset

VERBOSITY_LEVEL = 'WARNING'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


def _write_predict(idx, output_dir, prediction):
    """prediction should be list"""
    os.makedirs(output_dir, exist_ok=True)
    prediction = pd.Series(prediction, name='label')
    LOGGER.debug(f'prediction shape: {prediction.shape}')
    prediction.to_csv(
        join(output_dir, f'prediction_{idx}'), index=False, header=True)


def _predict(args):
    result = {}
    try:
        timer = Timer.from_file(join(args.temp_dir, 'timer.yaml'))
        LOGGER.info("===== Load test data")
        dataset = Dataset(args.dataset_dir)
        args.time_budget = dataset.get_metadata().get("time_budget")
        path.append(args.model_dir)
        LOGGER.info('==== Load user model')
        umodel = init_usermodel(dataset)
        with timer.time_limit('load'):
            umodel.load(args.temp_dir, timer.get_all_remain())

        LOGGER.info('==== start predicting')
        idx = args.idx
        y_preds = []
        while not dataset.is_end(idx):
            history = dataset.get_history(idx)
            pred_record = dataset.get_next_pred(idx)
            with timer.time_limit('predict', verbose=False):
                y_pred, next_step = umodel.predict(
                    history, pred_record, timer.get_all_remain())
            y_preds.extend(y_pred)
            idx += 1
            if next_step == 'update':
                result['is_end'] = False
                break
        else:
            result['is_end'] = True

        # Write predictions to output_dir
        _write_predict(idx, args.output_dir, y_preds)
        result = {
            **result,
            'idx': idx,
            'status': 'success',
            'next_step': next_step,
        }

        with timer.time_limit('save'):
            umodel.save(args.temp_dir, timer.get_all_remain())
        timer.save(join(args.temp_dir, 'timer.yaml'))

    except TimeoutException as ex:
        LOGGER.error(ex, exc_info=True)
        result['status'] = 'timeout'
    except Exception as ex:
        LOGGER.error(ex, exc_info=True)
        result['status'] = 'failed'

    return result


def _write_result(args, result):
    with open(args.result_file, 'w') as ftmp:
        yaml.dump(result, ftmp)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        help="Directory storing the dataset (containing "
                             "e.g. adult.data/)")
    parser.add_argument('--model_dir', type=str,
                        help="Directory storing the model ")

    parser.add_argument('--result_file', type=str,
                        help="a json file save the result")

    parser.add_argument('--temp_dir', type=str,
                        help="Directory storing the temporary output."
                             "e.g. save the participants` model "
                             "after trainning.")

    parser.add_argument('--output_dir', type=str,
                        help="Directory storing the predictions. It will "
                             "contain e.g. [start.txt, adult.predict_0, "
                             "adult.predict_1, ..., end.txt] when ingestion "
                             "terminates.")

    parser.add_argument("--idx", type=int, help="dataset idx")

    args = parser.parse_args()
    return args


def main():
    """main entry"""
    LOGGER.info('==== prediction process started')
    args = _parse_args()
    result = _predict(args)
    LOGGER.info('==== Write result')
    _write_result(args, result)


if __name__ == '__main__':
    main()
