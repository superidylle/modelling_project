# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from Tearsheet_Function import TearsheetStatistic
import config
pd.set_option('expand_frame_repr', False)

price_type = 'Adj Close'
start = "2014-03-01"
end = "2019-05-01"

symbol = ["MSFT", "AAPL", "HON", "XOM", "WMT", "PG","AMZN", "GOOG", "JPM", "MA",
          "VZ", "CSCO","UNH", "T", "PFE", "CVX", "HD", "MRK", "KO", "WFC", "BA",
          "INTC", "DIS", "CMCSA", "PEP", "ORCL", "NFLX", "MCD", "C", "LLY", "PM",
          "UNP", "MDT", "ABBV", "CRM", "IBM", "ACN", "AMGN", "UTX", "TMO", "TXN", "AXP", "NEE", "LMT", "MMM","SBUX"]

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

stocks = combine_stock_date(symbol, start, end)
stocks = stocks.set_index('Date')

etf = import_stock_data("SPY", start, end)
etf = etf.set_index('Date')

stocks_train = stocks[:50]
etf_train = etf[:50]
stocks_test = stocks[50:]
etf_test  = etf[50:]

isCoint = pd.DataFrame(np.zeros([1, len(symbol)], dtype=bool), columns=symbol, index=['isCoint'])

confidence = 0.9
if confidence == 0.95:
    X = 1
elif confidence == 0.99:
    X = 2
else:
    X = 0

for col in isCoint:

    y2 = etf_train.join(stocks_train[col]).dropna()
    result = coint_johansen(y2, 0, 1)
    # print(result.lr1)
    # print(result.cvt)
    if result.lr1[0] > result.cvt[0, X]:
        isCoint[col] = True

capital_allocation = np.tile(isCoint, [len(stocks_train), 1]) * stocks_train
capital_allocation = capital_allocation.replace(0, np.NaN)

log_market_value_train = pd.Series(np.sum(np.log(capital_allocation), axis=1), name='log_market_value')

portfolio_train = np.log(etf_train).join(log_market_value_train)

result = coint_johansen(portfolio_train, 0, 1)

if not (result.lr1[0] > result.cvt[0, 0]):
    print("The portfolio doesn't cointegrate")
    exit()

else:
    print("********* Continue *************")

    capital_allocation_test = (np.tile(isCoint, [len(stocks_test), 1]) * stocks_test).join(etf_test)
    capital_allocation_test =capital_allocation_test.replace(0, np.NaN)

    temp1 = np.tile(result.evec[1, 0], [len(stocks_test), len(isCoint.T)])
    temp2 = np.tile(result.evec[0, 0], [len(stocks_test), 1])
    weights = np.hstack((temp1, temp2))

    log_market_value_test = np.sum(np.log(capital_allocation_test) * weights, axis=1)

    lookback = 30
    moving_mean = log_market_value_test.rolling(lookback).mean()
    moving_std = log_market_value_test.rolling(lookback).std()

    number_units = pd.DataFrame(-(log_market_value_test - moving_mean) / moving_std)
    position = pd.DataFrame(np.tile(number_units, [1, len(weights.T)]) * weights)
    pnl = np.multiply(position[:-1], np.diff(np.log(capital_allocation_test), axis=0)).sum(axis=1)

    # make SPY data as benchmark
    data = capital_allocation_test[['SPY']]
    data.reset_index(drop=False, inplace=True)

    # calculate daily pnl and return
    pnl = pnl.shift(1)
    data['pnl'] = pnl
    gross_market_value = np.sum(position, axis=1)
    data['mrk_val'] = gross_market_value.shift(1)
    data['return'] = data['pnl'] / data['mrk_val']
    data = data.fillna(0)
    data['acum_rtn'] = np.cumsum(data['return'])

if __name__ == '__main__':

    data.set_index('Date', drop=True, inplace=True)
    test = TearsheetStatistic(data, title='SPY Component Strategy')
    test.save()