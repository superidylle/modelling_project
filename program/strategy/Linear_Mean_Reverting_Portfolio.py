#-*- coding: utf-8 -*-

"""
Johansen cointegration test
Input x, p, k
x = input matrix of time-series (number of observations * m)
p = order of time polynomial in the null-hypothesis
    p = -1,  no deterministic part
    p = 0, for constant term
    p = 1, for constant plus time-trend
    p > 1, for higher order polynomial
k = number of lagged differences

Returned Results:
result.eig = eigenvalues (m * 1)
result.evec = eigenvectors (m * m), first r columns are normalized coint vector
result.lr1 = likelihood ratio of trace statistics for r=0 to m-1 (m * 1)
result.lr2 = maximum eigenvalue statistics r=0 to m-1 (m * 1)
result.cvt = critical value for trace statistic (m * 3)
result.cvm = critical value for max eigen value statistics
result.ind = index of co-integrating variables ordered by size of the eigenvalues from large to small
"""

import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import scipy
import config
from Tearsheet_Function import TearsheetStatistic
pd.set_option('expand_frame_repr', False)

# calculate the Hurst Exponent
def hurst(ts):

    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]*2.0

# calculate the half life of mean reversion
def half_life(ts):
    """
    calculate the delta for each observation
    delta = p(t) - p(t-1)
    :param ts:
    :return: half_life value
    """
    delta_ts = np.diff(ts)
    # ts[1:] the vector of lagged prices, lag = 1
    # stack up a vector of ones and  transpose
    lag_ts = np.vstack([ts[1:], np.ones(len(ts[1:]))]).T
    # calculate the slope of the deltas vs the lagged values
    beta = scipy.linalg.lstsq(lag_ts, delta_ts)
    # calculate the half_life
    half_life = np.log(2) / beta[0]
    return half_life[0]

def import_stock_data(code,start, end):
    df = pd.read_csv(config.input_data_path + '/' + code + '.csv')
    df.rename(columns={'Adj Close': code}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by=['Date'], inplace=True)
    df = df[['Date', code]]
    df = df[ (df['Date'] >= start) & (df['Date'] <= end)]
    df.reset_index(inplace=True, drop=True)
    return df

def combine_stock_date(symbol, start, end):

    stock_data = pd.DataFrame(columns=['Date'])
    for code in symbol:
        df = import_stock_data(code, start, end)
        stock_data = pd.merge(left=df, right=stock_data, on=['Date'], how='left', sort=True, indicator=False)
        stock_data.fillna(method='ffill')
    return stock_data

class JohansenTestStrategy():

    def __init__(self, symbol_x, symbol_y, symbol_z, start, end):

        self.symbol_x = symbol_x
        self.symbol_y = symbol_y
        self.symbol_z = symbol_z
        self.start = start
        self.end = end
        self.stock_list = [self.symbol_x, self.symbol_y, self.symbol_z]
        self.data = self.benchmark_update()
        self.share_allocation = {}

    def benchmark_update(self):

        self.data = self.calculate_result()
        self.benchmark = import_stock_data('SPY', self.start, self.end)
        self.data = pd.merge(left=self.data, right=self.benchmark, on=['Date'], how='left', sort=True, indicator=False)
        self.data.set_index('Date', drop=True, inplace=True)
        return self.data

    def calculate_result(self):
        self.data = self.johansen_test_result()
        half_life_time = half_life(self.data['port'])
        lookback = int(half_life_time)
        self.data['moving_mean'] = self.data['port'].rolling(lookback).mean()
        self.data['moving_std'] = self.data['port'].rolling(lookback).std()
        self.data['z_score'] = (self.data['port'] - self.data['moving_mean']) / self.data['moving_std']

        # multiples of units portfolio
        self.data['number_units'] = self.data['z_score'] * -1
        AA = np.tile(self.data[['number_units']], (1, 3))
        BB = np.multiply(np.tile(self.share_allocation, (len(self.data), 1)), self.data[[self.symbol_x, self.symbol_y, self.symbol_z]])
        position =  pd.DataFrame(np.multiply(AA, BB))

        # result.evec is the shares allocation, positions is the capital(dollar) allocation in each ETF
        self.data[[self.symbol_x + '_pos', self.symbol_y+ '_pos', self.symbol_z + '_pos']] = position

        # calculate the porfit and loss
        self.data['pnl'] = np.multiply(position[:-1], np.divide(np.diff(self.data[[self.symbol_x, self.symbol_y, self.symbol_z]], axis=0), self.data[[self.symbol_x, self.symbol_y, self.symbol_z]][:-1])).sum(axis=1)
        self.data['mrk_val'] = pd.DataFrame.sum(abs(position), axis=1)
        self.data['return'] =  self.data['pnl'] / self.data['mrk_val']
        self.data['acum_rtn'] = pd.DataFrame(np.cumsum(self.data['return']))
        self.data['acum_rtn'] = self.data['acum_rtn'] .fillna(method='pad')
        return self.data

    def johansen_test_result(self):

        self.data = self.import_data()
        result = coint_johansen(self.data, 0, 1)
        self.share_allocation = result.evec[:, 0]
        self.data['port'] = pd.DataFrame.sum(self.share_allocation* self.data, axis=1)
        return self.data

    def import_data(self):

        self.data = combine_stock_date(self.stock_list, self.start, self.end)
        self.data = self.data.set_index('Date')
        return self.data

if __name__ == '__main__':

    symbol_x = "GOOG"
    symbol_y = "AAPL"
    symbol_z = "INTC"
    start = "2014-01-01"
    end = "2018-12-01"
    df = JohansenTestStrategy(symbol_x, symbol_y, symbol_z, start, end)
    test = TearsheetStatistic(df.data, title='Portfolio Linear Mean Reverting Strategy')
    test.save()
