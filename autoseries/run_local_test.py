"""run local test in starting kit"""
# pylint: disable=logging-fstring-interpolation

import argparse
import logging
import os
from os.path import join, isdir
import shutil
from multiprocessing import Process

VERBOSITY_LEVEL = 'WARNING'

logging.basicConfig(
    level=getattr(logging, VERBOSITY_LEVEL),
    format='%(asctime)s %(levelname)s %(filename)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def _here(*args):
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(here, *args)


def _ingestion_program(starting_kit_dir):
    return join(starting_kit_dir, 'ingestion', 'ingestion.py')


def _scoring_program(starting_kit_dir):
    return join(starting_kit_dir, 'scoring', 'score.py')


def remove_dir(output_dir):
    """Remove the directory `output_dir`.
  This aims to clean existing output of last run of local test.
  """
    if isdir(output_dir):
        logging.info(
            f"Cleaning existing output directory of last run: {output_dir}")
        shutil.rmtree(output_dir)


def _clean(starting_kit_dir):
    ingestion_output_dir = join(starting_kit_dir, 'sample_result_submission')
    score_dir = os.path.join(starting_kit_dir, 'scoring_output')
    remove_dir(ingestion_output_dir)
    remove_dir(score_dir)


def run(dataset_dir, code_dir):
    """run"""
    # Current directory containing this script
    starting_kit_dir = _here()
    path_ingestion = _ingestion_program(starting_kit_dir)
    path_scoring = _scoring_program(starting_kit_dir)

    # Run ingestion and scoring at the same time
    command_ingestion = (
        'python '
        #  f'{path_ingestion} --dataset_dir={dataset_dir}/data '
        f'{path_ingestion} --dataset_dir={dataset_dir}/train.data'
        f' --code_dir={code_dir}')

    command_scoring = (
        #  f'python {path_scoring} --solution_dir={dataset_dir}/solution')
        f'python {path_scoring} --solution_dir={dataset_dir}')

    def run_ingestion():
        os.system(command_ingestion)

    def run_scoring():
        os.system(command_scoring)

    ingestion_process = Process(name='ingestion', target=run_ingestion)
    scoring_process = Process(name='scoring', target=run_scoring)
    _clean(starting_kit_dir)

    ingestion_process.start()
    scoring_process.start()


def _parse_args():
    default_starting_kit_dir = _here()
    default_dataset_dir = join(default_starting_kit_dir, 'data', '15')
    default_code_dir = join(default_starting_kit_dir, 'code_submission')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default=default_dataset_dir,
                        help="Directory storing the dataset, should contain"
                             "'data' and 'solution'")

    parser.add_argument('--code_dir', type=str,
                        default=default_code_dir,
                        help="Directory storing the submission code "
                             "`model.py` and other necessary packages.")

    args = parser.parse_args()
    return args


def main():
    """main entry"""
    args = _parse_args()
    dataset_dir = args.dataset_dir
    code_dir = args.code_dir
    logging.info("#" * 50)
    logging.info("Begin running local test using")
    logging.info(f"code_dir = {code_dir}")
    logging.info(f"dataset_dir = {dataset_dir}")
    logging.info("#" * 50)
    run(dataset_dir, code_dir)


if __name__ == '__main__':
    main()
