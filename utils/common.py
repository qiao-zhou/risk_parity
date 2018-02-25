import numpy as np
import time
import logging


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        name = kw.get('log_name', method.__name__.upper())
        time_elapsed_in_min = np.round((te - ts) / 60, 2)
        logging.info("Running %s step takes %4.2f minutes" % (name, time_elapsed_in_min))
        return result

    return timed


def ret_annualizer(frequency="daily"):
    if frequency == "daily":
        return 252
    elif frequency == "monthly":
        return 12
    elif frequency == "quarterly":
        return 4


def vol_annualizer(frequency="daily"):
    if frequency == "daily":
        return np.sqrt(252)
    elif frequency == "monthly":
        return np.sqrt(12)
    elif frequency == "quarterly":
        return np.sqrt(4)
