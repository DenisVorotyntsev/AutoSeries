# pylint: disable=logging-fstring-interpolation, broad-except
"""ingestion program for autoWSL"""
import os
from os.path import join
import sys
from sys import path
import argparse
import datetime
import subprocess
import threading
import time
import numpy as np
import yaml
from filelock import FileLock

from common import get_logger, init_usermodel

import timing
from timing import Timer
from dataset import Dataset


# Verbosity level of logging:
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)

PROCESSES = ['train', 'predict', 'update', 'save', 'load']
PROCESSES_MODE = {
    'train': timing.RESET,
    'predict': timing.CUM,
    'update': timing.CUM,
    'save': timing.RESET,
    'load': timing.RESET
}

OP_MAP = {
    'mean': np.mean,
    'max': np.max,
    'std': np.std
}


def _here(*args):
    """Helper function for getting the current directory of this script."""
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(here, *args))


def write_start_file(output_dir):
    """Create start file 'start.txt' in `output_dir` with updated timestamp
    start time.

    """
    LOGGER.info('===== alive_thd started')
    start_filepath = os.path.join(output_dir, 'start.txt')
    lockfile = os.path.join(output_dir, 'start.txt.lock')
    while True:
        current_time = datetime.datetime.now().timestamp()
        with FileLock(lockfile):
            with open(start_filepath, 'w') as ftmp:
                yaml.dump(current_time, ftmp)
        time.sleep(10)


class IngestionError(RuntimeError):
    """Model api error"""


def _parse_args():
    root_dir = _here(os.pardir)
    default_dataset_dir = join(root_dir, "sample_data")
    default_output_dir = join(root_dir, "sample_result_submission")
    default_ingestion_program_dir = join(root_dir, "ingestion_program")
    default_code_dir = join(root_dir, "code_submission")
    default_score_dir = join(root_dir, "scoring_output")
    default_temp_dir = join(root_dir, 'temp_output')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default=default_dataset_dir,
                        help="Directory storing the dataset (containing "
                             "e.g. adult.data/)")
    parser.add_argument('--output_dir', type=str,
                        default=default_output_dir,
                        help="Directory storing the predictions. It will "
                             "contain e.g. [start.txt, adult.predict_0, "
                             "adult.predict_1, ..., end.yaml] when ingestion "
                             "terminates.")
    parser.add_argument('--ingestion_program_dir', type=str,
                        default=default_ingestion_program_dir,
                        help="Directory storing the ingestion program "
                             "`ingestion.py` and other necessary packages.")
    parser.add_argument('--code_dir', type=str,
                        default=default_code_dir,
                        help="Directory storing the submission code "
                             "`model.py` and other necessary packages.")
    parser.add_argument('--score_dir', type=str,
                        default=default_score_dir,
                        help="Directory storing the scoring output "
                             "e.g. `scores.txt` and `detailed_results.html`.")
    parser.add_argument('--temp_dir', type=str,
                        default=default_temp_dir,
                        help="Directory storing the temporary output."
                             "e.g. save the participants` model after "
                             "trainning.")

    args = parser.parse_args()
    LOGGER.debug(f'Parsed args are: {args}')
    LOGGER.debug("-" * 50)
    if (args.dataset_dir.endswith('run/input') and
            args.code_dir.endswith('run/program')):
        LOGGER.debug("Since dataset_dir ends with 'run/input' and code_dir "
                     "ends with 'run/program', suppose running on "
                     "CodaLab platform. Modify dataset_dir to 'run/input_data'"
                     " and code_dir to 'run/submission'. "
                     "Directory parsing should be more flexible in the code of"
                     " compute worker: we need explicit directories for "
                     "dataset_dir and code_dir.")

        args.dataset_dir = args.dataset_dir.replace(
            'run/input', 'run/input_data')
        args.code_dir = args.code_dir.replace(
            'run/program', 'run/submission')

        # Show directories for debugging
        LOGGER.debug(f"sys.argv = {sys.argv}")
        LOGGER.debug(f"Using dataset_dir: {args.dataset_dir}")
        LOGGER.debug(f"Using output_dir: {args.output_dir}")
        LOGGER.debug(
            f"Using ingestion_program_dir: {args.ingestion_program_dir}")
        LOGGER.debug(f"Using code_dir: {args.code_dir}")
    return args


def _init_python_path(args):
    path.append(args.ingestion_program_dir)
    path.append(args.code_dir)
    # IG: to allow submitting the starting kit as sample submission
    path.append(args.code_dir + '/sample_code_submission')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)


def _train(umodel, dataset, timer):
    # Train the model
    train_dataset = dataset.get_train()

    with timer.time_limit('train'):
        next_step = umodel.train(train_dataset, timer.get_all_remain())

    return next_step


def _update(umodel, idx, dataset, timer):
    train_data = dataset.get_train()
    test_historical_data = dataset.get_all_history(idx)
    with timer.time_limit('update'):
        next_step = umodel.update(train_data, test_historical_data,
                                  timer.get_all_remain())
    return next_step


def _predict(umodel, args, idx, timer):
    # Make predictions using the trained model
    result_file = join(args.temp_dir, 'predproc_result.yaml')
    timer_file = join(args.temp_dir, 'timer.yaml')
    predict_py = join(
        os.path.dirname(os.path.realpath(__file__)), 'predict.py')

    with timer.time_limit('save'):
        umodel.save(args.temp_dir, timer.get_all_remain())
    timer.save(timer_file)

    LOGGER.info("===== call subprocess of prediction")
    subprocess.run(
        f"python {predict_py} --dataset_dir {args.dataset_dir} "
        f"--model_dir {args.code_dir} --result_file {result_file} "
        f"--output_dir {args.output_dir} --temp_dir {args.temp_dir} "
        f"--idx {idx}", shell=True, check=True)

    with open(result_file, 'r') as ftmp:
        result = yaml.safe_load(ftmp)
    timer.load(timer_file)
    with timer.time_limit('load'):
        umodel.load(args.temp_dir, timer.get_all_remain())

    if result['status'] == 'timeout':
        raise IngestionError('predicting timeout')
    elif result['status'] != 'success':
        raise IngestionError('error occurs when predicting')

    return result['idx'], result['next_step'], result['is_end']


def _finalize(args, timer, n_update):
    # Finishing ingestion program
    end_time = time.time()

    time_stats = timer.get_all_stats()
    for pname, stats in time_stats.items():
        for stat_name, val in stats.items():
            LOGGER.info(f'the {stat_name} of duration in {pname}: {val} sec')

    overall_time_spent = timer.get_overall_duration()

    # Write overall_time_spent to a end.yaml file
    end_filename = 'end.yaml'
    content = {
        'ingestion_duration': overall_time_spent,
        'time_stats': time_stats,
        'n_update': n_update,
        'end_time': end_time}

    with open(join(args.output_dir, end_filename), 'w') as ftmp:
        yaml.dump(content, ftmp)
        LOGGER.info(
            f'Wrote the file {end_filename} marking the end of ingestion.')

        LOGGER.info("[+] Done. Ingestion program successfully terminated.")
        LOGGER.info(f"[+] Overall time spent {overall_time_spent:5.2} sec")
        LOGGER.info(f"[+] Overall update used: {n_update}")

    # Copy all files in output_dir to score_dir
    os.system(
        f"cp -R {os.path.join(args.output_dir, '*')} {args.score_dir}")
    LOGGER.debug(
        "Copied all ingestion output to scoring output directory.")

    LOGGER.info("[Ingestion terminated]")


def _init_timer(time_budgets):
    timer = Timer()
    for process in PROCESSES:
        timer.add_process(
            process, time_budgets[process], PROCESSES_MODE[process])
        LOGGER.debug(
            f"init time budget of {process}: {time_budgets[process]} "
            f"mode: {PROCESSES_MODE[process]}")
    return timer


def main():
    """main entry"""
    LOGGER.info('===== Start ingestion program.')
    # Parse directories from input arguments
    LOGGER.info('===== Initialize args.')
    args = _parse_args()
    _init_python_path(args)

    LOGGER.info('===== Set alive_thd')
    alive_thd = threading.Thread(target=write_start_file, name="alive",
                                 args=(args.output_dir,))
    alive_thd.daemon = True
    alive_thd.start()

    LOGGER.info('===== Load data.')
    dataset = Dataset(args.dataset_dir)
    args.time_budget = dataset.get_metadata().get("time_budget")

    for key, value in args.time_budget.items():
        LOGGER.info(f"Time budget for {key}: {value}")

    LOGGER.info("===== import user model")
    umodel = init_usermodel(dataset)

    LOGGER.info("===== Begin training user model")
    timer = _init_timer(args.time_budget)

    next_step = _train(umodel, dataset, timer)
    idx = 0
    n_update = 0
    while True:
        if next_step == 'predict':
            LOGGER.info("===== predict")
            idx, next_step, is_end = _predict(umodel, args, idx, timer)
            if is_end:
                break
        elif next_step == 'update':
            n_update += 1
            LOGGER.info(f"===== update ({n_update})")
            next_step = _update(umodel, idx, dataset, timer)
        else:
            raise IngestionError(
                f"wrong next_step [{next_step}], "
                "should be {predict, update}")

    _finalize(args, timer, n_update)


if __name__ == "__main__":
    main()
