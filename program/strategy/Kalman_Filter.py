# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import config
import matplotlib.pyplot as plt
from Tearsheet_Function import TearsheetStatistic
pd.set_option('expand_frame_repr', False)

def import_data(code):
    df = pd.read_csv(config.input_data_path + '/' + code + '.csv')
    df = df[['Date', 'Adj Close']]
    df.rename(columns={'Adj Close': code}, inplace=True)
    df.sort_values(by=['Date'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.reset_index(inplace=True, drop=True)
    return df

# import data
data = pd.read_csv(config.input_data_path + '/' + 'EWA EWC.csv', parse_dates=True)
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(by=['Date'], inplace=True)
data.set_index('Date', drop=True, inplace=True)

x_ticket = 'EWA'
y_ticket = 'EWC'

x = pd.DataFrame(data[x_ticket])
y = np.array(data[y_ticket])
index = x.index

x['ones'] = np.ones(len(x))
x = np.asarray(x)

delta = 0.0001

yhat = np.zeros(shape=(len(x), 1))
e = np.zeros(shape=(len(x), 1))
Q = np.zeros(shape=(len(x), 1))

R = np.zeros((2,2))
P = np.zeros((2,2))
beta = np.zeros((2, len(x)))
Vw = delta/(1-delta) * np.eye(2)
Ve = 0.001

for t in range(len(y)):
    if t > 0:
        beta[:,t] = beta[:,t-1]
        R = P + Vw
        yhat[t] = x[t, :].dot(beta[:, t])
        Q[t] = x[t, :].dot(R).dot(x[t,:].T) + Ve

        e[t] = y[t] - yhat[t]
        K = R.dot(x[t, :].T) / Q[t]
        beta[:,t] = beta[:, t] + K * e[t]
        P = R - K * x[t, :] * R

sqrt_Q = np.sqrt(Q)
beta = pd.DataFrame(beta.T, index=index, columns=('x', 'intercept'))
e = pd.DataFrame(e, index=index)

data[['beta', 'intercept']] = beta
data['sqrt_Q'] = sqrt_Q
data[['e']] = e

long_entry = e < -sqrt_Q  # a long position means we should buy EWC
long_exit = e > -sqrt_Q
short_entry = e > sqrt_Q
short_exit = e < sqrt_Q

numunits_long = np.zeros((len(data), 1))
numunits_long = pd.DataFrame(np.where(long_entry, 1, 0))
numunits_short = np.zeros((len(data), 1))
numunits_short = pd.DataFrame(np.where(short_entry, -1, 0))
numunits = numunits_long + numunits_short

data.reset_index(inplace=True, drop=False)
data['number_units'] = numunits


# compute the position for each asset
AA = pd.DataFrame(np.tile(numunits, (1, 2)))
BB = pd.DataFrame(-beta['x'])
BB['ones'] = np.ones((len(beta)))
data[[x_ticket+'_pos', y_ticket+'_pos']] = np.multiply(np.multiply(AA, BB), data[[x_ticket, y_ticket]])

# compute the daily pnl in $$
data['pnl']  = np.multiply(data[[x_ticket+'_pos', y_ticket+'_pos']][:-1], np.divide(np.diff(data[[x_ticket, y_ticket]], axis=0), data[[x_ticket,y_ticket]][:-1])).sum(axis=1)

# gross market value of portfolio
mrk_val = pd.DataFrame.sum(abs(data[[x_ticket+'_pos', y_ticket+'_pos']]), axis=1)
data['mrk_val'] = mrk_val

# return is P&L divided by gross market value of portfolio
data['return'] = data['pnl'] / data['mrk_val']
data['return'] = data['return'].fillna(0)
data['equity'] = (data['return'] + 1).cumprod()
data['equity'][0] = 1
data['equity'] = data['equity'].fillna(method='ffill')

# cumulative return and smoothing of series for plotting
data['acum_rtn'] = pd.DataFrame(np.cumsum(data['return']))
data['acum_rtn'] = data['acum_rtn'].fillna(method='pad')

# add benchmark
benchmark = import_data('SPY')
data = pd.merge(left=data, right=benchmark, on=['Date'], how='left', sort=True, indicator=False)

#plotting the chart
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data['e'], color='blue')
ax.plot(sqrt_Q, color='black')
ax.plot(-sqrt_Q, color='black')

ax.set_title('error vs sqrt(variance prediction)')
ax.set_xlabel('Data points')
ax.set_ylabel('# std')
ax.set_ylim(-2, 2)
plt.show()

if __name__ == '__main__':
    data.set_index('Date', drop=True, inplace=True)
    test = TearsheetStatistic(data, title='Kalman Filter Strategy')
    test.save()





