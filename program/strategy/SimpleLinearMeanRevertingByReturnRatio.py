# -*- coding: utf-8 -*-

import pandas as pd
from pyfinance.ols import OLS, RollingOLS, PandasRollingOLS
from Tearsheet_Function import TearsheetStatistic
import config
import numpy as np


def import_data(code,start, end):
    df = pd.read_csv(config.input_data_path + '/' + code + '.csv')
    df.rename(columns={'Adj Close': code}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by=['Date'], inplace=True)
    df = df[['Date', code]]
    df = df[ (df['Date'] >= start) & (df['Date'] <= end)]
    df.reset_index(inplace=True, drop=True)
    return df

start = "2011-08-01"
end = "2017-12-01"

y_ticker = 'AAPL'
x_ticker = 'MSFT'

df1 = import_data( x_ticker, start, end)
df2 = import_data(y_ticker, start, end)

data = pd.merge(left=df1, right=df2, on='Date', how='right', sort=True, indicator=False)
data.fillna(method='ffill', inplace=True)
data.reset_index(drop=True, inplace=True)

y = data[y_ticker]
x = data[x_ticker]

lookback = 20

ratio = y/x

moving_mean = ratio.rolling(lookback).mean()
moving_std = ratio.rolling(lookback).std()
number_units = pd.DataFrame(-(ratio - moving_mean) / moving_std)
AA = pd.DataFrame(np.tile(number_units, (1, 2)))
BB = np.tile([-1, 1], (len(number_units), 1))
data = data.set_index('Date')
data[[x_ticker + '_pos', y_ticker + '_pos']] = pd.DataFrame(np.multiply(data, np.multiply(AA, BB)))

data['pnl'] = np.multiply(data[[x_ticker + '_pos', y_ticker + '_pos']][:-1], np.divide(np.diff(data[[x_ticker, y_ticker]], axis=0), data[[x_ticker, y_ticker]][:-1])).sum(axis=1)

data['mrk_val'] = pd.DataFrame.sum(np.abs(data[[x_ticker + '_pos', y_ticker + '_pos']]), axis=1)

data['return'] = data['pnl'] / data['mrk_val']
data = data.fillna(0)
data['acum_rtn'] = np.cumsum(data['return'])

benchmark = import_data('SPY', start, end)
data = pd.merge(left=data, right=benchmark, on=['Date'], how='left', sort=True, indicator=False)

if __name__ == '__main__':
    data.set_index('Date', drop=True, inplace=True)
    test = TearsheetStatistic(data, title='Simple Linear Mean Reverting by return ratio Strategy')
    test.save()
