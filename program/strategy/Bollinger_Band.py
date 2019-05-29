# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from program import config
from Tearsheet_Function import TearsheetStatistic
from pyfinance.ols import PandasRollingOLS
pd.set_option('expand_frame_repr', False)

def import_data(code):
    df = pd.read_csv(config.input_data_path + '/' + code + '.csv')
    df = df[['Date', 'Adj Close']]
    df.rename(columns={'Adj Close': code}, inplace=True)
    df.sort_values(by=['Date'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.reset_index(inplace=True, drop=True)
    return df

class BollingerBandStrategy():

    def __init__(self, x_ticker, y_ticker, star, end, entry_Zscore=1, exit_Zscore=0, lookback=20):
        self.x_ticker = x_ticker
        self.y_ticker = y_ticker
        self.star = star
        self.end = end
        self.entry_Zscore = entry_Zscore
        self.exit_Zscore = exit_Zscore
        self.lookback = lookback
        self.data = self.benchmark_update()

    def benchmark_update(self):
        self.data = self.calculate_accumulate_return()
        self.benchmark = import_data('SPY')
        self.data = pd.merge(left=self.data, right=self.benchmark, on=['Date'], how='left', sort=True, indicator=False)
        self.data.set_index('Date', drop=True, inplace=True)
        return self.data

    def calculate_accumulate_return(self):
        self.data = self.calculate_return()
        self.data['acum_rtn'] = np.cumsum(self.data['return']).fillna(method='pad')
        return self.data

    def calculate_return(self):
        self.data = self.calculate_market_value()
        self.data['return'] = self.data['pnl'] / self.data['mrk_val']
        self.data['return'] = self.data['return'].fillna(0)
        self.data['equity'] = (self.data['return'] + 1).cumprod()
        self.data['equity'][0] = 1
        self.data['equity'] = self.data['equity'].fillna(method='ffill')

        return self.data

    def calculate_market_value(self):
        position = self.number_holding()
        self.data = self.calculate_pnl()
        self.data['mrk_val'] = pd.DataFrame.sum(abs(position), axis=1)
        self.data['mrk_val'] = self.data['mrk_val'].shift(1)
        return self.data

    def calculate_pnl(self):
        position = self.number_holding()
        data = self.data[[self.y_ticker, self.x_ticker]]
        pnl = np.multiply(position[:-1], np.divide(np.diff(data, axis=0), data[:-1])).sum(axis=1)
        pnl = pd.Series([0]).append(pnl)
        pnl = pnl.reset_index(drop=True)
        self.data['pnl'] = pnl
        return self.data

    def number_holding(self):
        self.data = self.Z_score_calulation()

        long_entry = self.data['Zscore'] < - self.entry_Zscore
        long_exit = self.data['Zscore'] >= - self.exit_Zscore
        short_entry = self.data['Zscore'] > self.entry_Zscore
        short_exit = self.data['Zscore'] <= self.exit_Zscore

        number_long = np.zeros((len(self.data['port']), 1))
        number_long = pd.DataFrame(np.where(long_entry, 1, 0))
        number_short = np.zeros((len(self.data['port']), 1))
        number_short = pd.DataFrame(np.where(short_entry, -1, 0))

        number_holding = number_short + number_long

        AA = pd.DataFrame(np.tile(number_holding, (1, 2)))
        BB = pd.DataFrame(-self.data['betas'])
        BB['ones'] = np.ones((len(self.data['betas'])))
        order = ['ones', 'betas']
        BB = BB[order]
        position = np.multiply(np.multiply(AA, BB), self.data[[self.y_ticker, self.x_ticker]])
        position.columns = [self.y_ticker + 'position', self.x_ticker + 'position']
        self.data = self.data.reset_index(drop=True)
        self.data = pd.concat([self.data, position], axis=1)
        return position

    def Z_score_calulation(self):
        self.data = self.market_value_calculation()
        moving_mean = self.data['port'].rolling(self.lookback).mean()
        moving_std = self.data['port'].rolling(self.lookback).std()
        Zscore = (self.data['port'] - moving_mean) / moving_std
        self.data['Zscore'] = Zscore
        return self.data

    def market_value_calculation(self):
        self.data = self.beta_calculation()
        self.data['port'] = self.data.apply(lambda x: x[self.y_ticker] - x['betas'] * x[self.x_ticker], axis=1)
        return self.data

    def beta_calculation(self):
        self.data = self.import_data()

        model0 = PandasRollingOLS(y=self.data[self.y_ticker], x=self.data[self.x_ticker], window=self.lookback)
        self.data = self.data[self.lookback-1:]
        self.data['betas'] = model0.beta
        return self.data

    def import_data(self):
        self.x_data = import_data(self.x_ticker)
        self.y_data = import_data(self.y_ticker)
        self.data = pd.merge(left=self.y_data, right=self.x_data, on='Date', how='right', sort=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.set_index('Date', drop=True, inplace=True)
        self.data = self.data[self.star: self.end]
        self.data.reset_index(drop=False, inplace=True)
        self.x_data = self.data[self.x_ticker]
        self.y_data = self.data[self.y_ticker]
        return self.data

if __name__ == '__main__':

    df = BollingerBandStrategy('GLD', 'USO', '2006-05-25', '2012-04-09')
    test = TearsheetStatistic(df.data, title='Bollinger Band Strategy')
    test.save()











