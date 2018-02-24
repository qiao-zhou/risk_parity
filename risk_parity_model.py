import os
import pandas as pd
import numpy as np

from utils.common import ret_annualizer, vol_annualizer


class RiskParityModel:
    def __init__(self,
                 tickers=None,
                 lookback=60,
                 correlation_blind=True):
        if tickers is None:
            tickers = ['SPY', 'AGG']
        self.data_root = os.getcwd()
        self.file_name = 'ETFs.csv'
        self.file_path = os.path.join(self.data_root, self.file_name)
        self.tickers = tickers
        self.lookback = lookback
        self.correlation_blind = correlation_blind
        self.frequency = "daily"
        print('initiating risk parity model')

    def gather_data(self, date_col = "date"):
        """
        prepare asset pricing data for risk parity model
        :return: asset return time series
        """
        df = pd.read_csv(self.file_path, usecols=[date_col] + self.tickers)
        df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d')
        df.dropna(inplace=True)
        df.set_index('date', inplace=True)
        df = (df.unstack()
              .to_frame("price")
              .reset_index()
              .rename(columns={"level_0": "asset_id"}))
        return df

    def calc_return(self, df):
        """
        calculates asset returns, df contains 'asset_id', 'date', 'price'
        :param df: historical pricing data
        :return:
        """
        asset_ret = (df.pivot_table(index='date', values='price', columns='asset_id')
                     .pct_change()
                     .unstack()
                     .to_frame('asset_return')
                     .reset_index())
        df = df.merge(asset_ret, on=['date', 'asset_id'], how='left')
        return df

    def calc_portf_return(self, df):
        """
        df should contain daily asset returns and daily weights
            date, asset_id, weight, asset_return
        :param df:
        :return:
        """
        portf_return = (df
                        .groupby("date")
                        .apply(lambda x: (x["weights"] * x["asset_return"]).sum())
                        .to_frame("port_ret"))
        return portf_return

    def calc_volatility(self, df):
        """
        calculate rolling historical volatility
        :param df:
        :return:
        """
        vol_pt = (df.pivot_table(values='asset_return', index='date', columns='asset_id')
                      .rolling(window=self.lookback).std()*vol_annualizer(self.frequency))
        vol = vol_pt.unstack().to_frame('vol').reset_index()
        df = df.merge(vol, on=['date', 'asset_id'], how='left')
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
        if self.correlation_blind:
            df['weights'] = 1 / df['vol']
            df['weights'] = (df[['asset_id', 'date', 'weights']]
                             .groupby('date')['weights']
                             .transform(lambda x: x / x.sum()))
        else:
            raise NotImplementedError

        return df

    def calc_risk_contribution(self, df):
        """
        calculates asset-wise risk contribution
        :param df:
        :return:
        """
        return df

    def analyze_performance(self, df):
        """
        summarize portfolio performance
        :param df:
        :return:
        """
        return df

    def run_backtest(self):
        """
        run everything, generate the entire backtest and analytics;
        :return:
        """
        df = self.gather_data()
        df = self.calc_return(df)
        df = self.calc_volatility(df)
        df = self.calc_weights(df)
        portf_return = self.calc_portf_return(df)
        portf_return.cumsum().plot()


if __name__ == '__main__':
    model = RiskParityModel()
    print(model.tickers)

    model.run_backtest()
