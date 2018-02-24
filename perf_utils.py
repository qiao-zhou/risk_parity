import numpy as np
import pandas as pd


def get_maxdd(rets):
    '''
    It works for numpy array as well as data frame
    '''
    xs = rets.cumsum()
    i = np.argmax(np.maximum.accumulate(xs) - xs)  # end of the period
    j = np.argmax(xs[:i])  # start of period
    dd_start = j
    dd_end = i
    mdd = xs[i] - xs[j]
    return (mdd, dd_start, dd_end)


def get_sharpe_ratio(rets, prds_in_yr=252, mean_mode="gm"):
    if mean_mode == "gm":
        m = ((1 + rets).prod()) ** (1.0 / len(rets)) - 1
    elif mean_mode == "am":
        m = np.mean(rets)
    # am = np.mean(1+rets)
    return m * np.sqrt(prds_in_yr) / np.std(rets)


def get_perf_metrics(rets, prds_in_yr=252, mean_mode="gm"):
    '''
    Get performance metrics for a single strategy
    '''
    ann_ret = np.mean(rets) * prds_in_yr
    ann_vol = np.std(rets) * np.sqrt(prds_in_yr)
    sharpe = get_sharpe_ratio(rets, prds_in_yr, mean_mode=mean_mode)
    (mdd, dd_start, dd_end) = get_maxdd(rets)
    var = np.percentile(rets, 5)
    tail_rets = pd.Series(rets)[rets < var]
    expected_short_fall = np.mean(tail_rets)
    tail_risk = np.std(tail_rets - expected_short_fall)

    return (ann_ret, ann_vol, sharpe, mdd, dd_start, dd_end, -var * np.sqrt(prds_in_yr),
            -expected_short_fall * np.sqrt(prds_in_yr), tail_risk * np.sqrt(prds_in_yr))


def get_perf_summary(df_ret, prds_in_yr=252, mean_mode="gm"):
    '''
    Get performance summary given a data frame of return series. 
    Each column represents returns for a(n) asset/strategy/portfolio. 
    '''
    df_perf = pd.DataFrame(index=df_ret.columns.values)
    df_perf["Ann_Ret"] = np.ones(len(df_ret.columns.values)) * np.nan
    df_perf["Ann_Vol"] = np.ones(len(df_ret.columns.values)) * np.nan
    df_perf["Sharpe"] = np.ones(len(df_ret.columns.values)) * np.nan
    df_perf["MaxDD"] = np.ones(len(df_ret.columns.values)) * np.nan
    # df_perf["MDD_Duration"] = np.ones(len(df_ret.columns.values)) * np.nan
    df_perf["DD_Start"] = np.ones(len(df_ret.columns.values)) * np.nan
    df_perf["DD_End"] = np.ones(len(df_ret.columns.values)) * np.nan
    df_perf["Var(95)"] = np.ones(len(df_ret.columns.values)) * np.nan
    df_perf["ES(95)"] = np.ones(len(df_ret.columns.values)) * np.nan
    df_perf["TR(95)"] = np.ones(len(df_ret.columns.values)) * np.nan

    for start_name in df_ret.columns.values:
        (ann_ret, ann_vol, sharpe, mdd, dd_start, dd_end, var, expected_short_fall, tail_risk) = get_perf_metrics(
            df_ret[start_name], prds_in_yr=prds_in_yr, mean_mode=mean_mode)
        df_perf.ix[start_name, :] = (
        ann_ret * 100, ann_vol * 100, sharpe, mdd * 100, dd_start, dd_end, var * 100, expected_short_fall * 100,
        tail_risk * 100)

    df_perf["DD_Start"] = [df_perf["DD_Start"][i].strftime("%Y-%m-%d") for i in range(len(df_perf["DD_Start"]))]
    df_perf["DD_End"] = [df_perf["DD_End"][i].strftime("%Y-%m-%d") for i in range(len(df_perf["DD_End"]))]
    return df_perf
