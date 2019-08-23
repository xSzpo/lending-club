import logging 
import feather
import sys
import pandas as pd
import datetime
import time
import os
from pathlib import Path


logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

log = logging.getLogger(__name__)

data_DIR = Path('/Users/xszpo/Google Drive/DataScience/Projects/201908_credit/'
                '01_data')


def timeit(method):
    def timed(*args, **kw):
        start_time = datetime.datetime.now()
        result = method(*args, **kw)
        time_elapsed = datetime.datetime.now() - start_time
        log.info('Function "{}" - time elapsed (hh:mm:ss.ms) {}'.format(
            method.__name__, time_elapsed))
        return result
    return timed


@timeit
def read_train():
    return feather.read_dataframe(
        os.path.join(data_DIR, 'DS_loans_IN_train.feather'))


if __name__ == "__main__":4
    X = read_train()
    log.info(X.shape[0])
    log.info(X.shape[1])

# tbc...