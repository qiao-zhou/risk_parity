import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from utils.common import vol_annualizer, timeit
from utils.performance_utils import get_basic_summary
import logging


class RiskParityModel:
    def __init__(self,
                 tickers=None,
                 sigma_lookback=60,
                 risk_contribution_lookback=None,
                 correlation_blind=True,
                 weights_prior=None,
                 shrinkage_factor=0.0,
                 shrink_weights=False,
                 asset_id_col="asset_id",
                 date_col="date",
                 weights_col="optimal_weights",
                 asset_ret_col="asset_return"):
        """
        :param tickers:
        :param sigma_lookback:  window used for computing historical volatility and covariance matrix. 
        :param risk_contribution_lookback: window used for calculating risk contribution. Default to sigma_lookback.
        :param correlation_blind:
        :param weights_prior: the default prior is to weight asset equally.
        :param shrink_weights: whether to shrink weights to prior
        :param shrinkage_factor: how aggressive you want to shrink the weights.
            shrinkage_factor = 1 will force the weights to prior;
            shrinkage_factor = 0 will apply no effective shrinkage.
        """
        if tickers is None:
            tickers = ['SPY', 'AGG']
        if weights_prior is None:
            weights_prior = np.array([1 / len(tickers)] * len(tickers))
        if risk_contribution_lookback is None:
            risk_contribution_lookback = sigma_lookback
        self.data_root = os.getcwd()
        self.tickers = tickers
        self.sigma_lookback = sigma_lookback
        self.risk_contribution_lookback = risk_contribution_lookback
        self.correlation_blind = correlation_blind
        self.rebal_frequency = "daily"
        self.shrink_weights = shrink_weights
        self.weights_prior = weights_prior
        self.shrinkage_factor = shrinkage_factor
        self.asset_id_col = asset_id_col
        self.date_col = date_col
        self.weights_col = weights_col
        self.asset_ret_col = asset_ret_col
        self.data_in_dir = os.path.join(self.data_root, "data_in")
        self.data_out_dir = os.path.join(self.data_root, "data_out")
        self.file_name = 'ETFs.csv'
        self.file_name_pricing_data = os.path.join(self.data_in_dir, self.file_name)

        print('initiating risk parity model')

    @property
    def index_cols(self):
        return [self.date_col, self.asset_id_col]

    def gather_data(self):
        """
        prepare asset pricing data for risk parity model
        :return: asset return time series
        """
        df = pd.read_csv(self.file_name_pricing_data,
                         usecols=[self.date_col] + self.tickers)
        df[self.date_col] = pd.to_datetime(df[self.date_col], format='%Y%m%d')
        df.dropna(inplace=True)
        df.set_index(self.date_col, inplace=True)
        df = (df.unstack()
              .to_frame("price")
              .reset_index()
              .rename(columns={"level_0": self.asset_id_col}))
        return df

    def calc_return(self, df):
        """
        calculates asset returns, df contains 'asset_id', 'date', 'price'
        :param df: historical pricing data
        :return:
        """
        asset_ret = (df.pivot_table(index=self.date_col, values='price', columns=self.asset_id_col)
                     .pct_change()
                     .unstack()
                     .to_frame(self.asset_ret_col)
                     .reset_index())
        df = df.merge(asset_ret, on=self.index_cols, how='left')
        return df

    def calc_portf_return(self, df):
        """
        df should contain daily asset returns and daily weights
            date, asset_id, weight, asset_return
        :param df:
        :return:
        """
        portf_return = (df
                        .groupby(self.date_col)
                        .apply(lambda x: (x[self.weights_col] * x[self.asset_ret_col]).sum())
                        .to_frame("port_ret"))
        return portf_return

    def calc_volatility(self, df):
        """
        calculate rolling historical volatility
        :param df:
        :return:
        """
        logging.debug("Calculating vol using %d day lookback" % (self.sigma_lookback))
        vol_pt = (df.pivot_table(values=self.asset_ret_col, index=self.date_col, columns=self.asset_id_col)
                  .rolling(window=self.sigma_lookback).std() * vol_annualizer(self.rebal_frequency))
        vol = vol_pt.unstack().to_frame('vol').reset_index()
        df = df.merge(vol, on=self.index_cols, how='left')
        return df

    def calc_weights(self, df):
        """
        calculate risk parity weight
        If correlation blind is True, compute weights using Naive Risk Parity
            where we assumed correlation among assets are 0
        Otherwise, compute weights using Real Risk Parity, considering the full covariance matrix
            where we try to equalize risk contribution among assets
        :param df:
        :return:
        """
        # to calculate the optimal weights with or without considering asset correlations
        if self.correlation_blind:
            df[self.weights_col] = 1 / df['vol']
            # normalize weights to unit sum
            df[self.weights_col] = (df[self.index_cols + [self.weights_col]]
                                    .groupby(self.date_col)[self.weights_col]
                                    .transform(lambda x: x / x.sum()))
            logging.debug("Calculating weights assuming zero asset correlations.")
        else:
            raise NotImplementedError

        logging.info("Mean asset weights prior to shrinkage= \n%s" %
                     (df[[self.asset_id_col, self.weights_col]].groupby(self.asset_id_col).mean()))
        pt_wts = (df[[self.asset_id_col, self.date_col, self.weights_col]]
                  .pivot_table(index=self.date_col, values=self.weights_col, columns=self.asset_id_col))

        pt_wts.to_csv(os.path.join(self.data_out_dir, "optimal_weights.csv"))
        if self.shrink_weights:
            pt_wts_shrunken = pt_wts.apply(lambda x: x + (self.weights_prior - x) * self.shrinkage_factor, axis=1)
            df_wts_shrunken = (pt_wts_shrunken.unstack().to_frame(self.weights_col).reset_index())
            df = df.drop(self.weights_col, 1).merge(df_wts_shrunken, on=self.index_cols, how="left")
            pt_wts_shrunken.to_csv(os.path.join(self.data_out_dir, "optimal_weights_shrunken.csv"))
            logging.debug("Shrinking weights towards prior: %s" % (self.weights_prior))
            logging.info("Mean asset weights post shrinkage = \n%s" %
                         (df[[self.asset_id_col, self.weights_col]].groupby(self.asset_id_col).mean()))

        return df

    @timeit
    def calc_risk_contribution(self, df):
        """
        Calculates asset-wise risk contribution using a rolling window.
        risk contribution is defined as: (weights_vect * (Sigma.dot(weights_vect))) / var_p
        :param df:
        :return:
        """
        weights = df.pivot_table(values=self.weights_col, index=self.date_col, columns=self.asset_id_col)
        pct_return = df.pivot_table(values=self.asset_ret_col, index=self.date_col, columns=self.asset_id_col)
        assert len(weights) == len(pct_return), "dimension mismatch between weights and return!"

        pct_risk_contrib = weights * 0
        for i in range(len(weights)):
            if i >= self.risk_contribution_lookback:
                weights_vect = weights.iloc[i]
                Sigma = pct_return.iloc[i - self.risk_contribution_lookback:i, :].cov()
                var_p = weights_vect.dot(Sigma).dot(weights_vect)
                risk_contribution_vect = (weights_vect * (Sigma.dot(weights_vect))) / var_p
                pct_risk_contrib.iloc[i, :] = np.array(risk_contribution_vect)
        logging.info("Mean risk contribution:\n %s" % (pct_risk_contrib.mean()))
        df = df.merge(pct_risk_contrib.unstack().to_frame("risk_contribution").reset_index(),
                      on=self.index_cols, how="left")

        return df

    @timeit
    def analyze_performance(self, portf_return):
        """
        summarize portfolio performance
        :param df:
        :return:
        """
        smry = get_basic_summary(portf_return)
        print(smry)

    @timeit
    def run_backtest(self):
        """
        run everything, generate the entire backtest and analytics;
        :return:
        """
        df = self.gather_data()
        df = self.calc_return(df)
        df = self.calc_volatility(df)
        df = self.calc_weights(df)
        df = self.calc_risk_contribution(df)
        portf_return = self.calc_portf_return(df)
        self.analyze_performance(portf_return)

        logging.info("run_backtest completed successfully.")
        # portf_return.cumsum().plot()
        # plt.show()


if __name__ == '__main__':
    logging.basicConfig(filename='RiskParityModel.log',
                        format='%(levelname)s:%(asctime)s:%(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG)

    tickers = ["SPY", "AGG", "GLD"]
    tickers = ["AGG","TLT","GLD","SPY","QQQ","IWM","EEM"]
    weights_prior = np.array([1 / len(tickers)] * len(tickers))
    model = RiskParityModel(
        tickers=tickers,
        sigma_lookback=60,
        correlation_blind=True,
        weights_prior=weights_prior,
        shrinkage_factor=0.25,
        shrink_weights=True,
    )
    print(model.tickers)

    model.run_backtest()
