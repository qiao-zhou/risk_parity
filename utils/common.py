import numpy as np


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
