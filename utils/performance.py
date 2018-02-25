import pandas as pd
import numpy as np
from utils.common import ret_annualizer, vol_annualizer


def get_basic_summary(df_return, frequency="daily"):
    """
    Generate a basic performance summary.
    :param df_return:
    :param frequency: daily, monthly or quarterly
    :return:
    """
    # average annual return
    ann_return = df_return.mean() * ret_annualizer(frequency=frequency)
    # volatility of all assets&portfolios
    ann_vol = df_return.std() * vol_annualizer(frequency=frequency)
    sharpe = ann_return / ann_vol
    summary = pd.DataFrame({"ann_return (%)": ann_return * 1e2,
                            "ann_vol (%)": ann_vol * 1e2,
                            "sharpe": sharpe}).round(2)
    return summary
