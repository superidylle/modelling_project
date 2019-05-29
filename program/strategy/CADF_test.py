# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import config
import statsmodels.formula.api as smf
from pyfinance.ols import OLS, RollingOLS, PandasRollingOLS
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = pd.read_csv(config.input_data_path + '/' + 'EWA EWC' + '.csv', index_col='Date')

x = data['EWA']
y = data['EWC']

x_ticker = 'EWA'
y_ticker = 'EWC'

k = np.polyfit(x, y, 1)
xx = np.linspace(min(x), max(x), 1000)
yy = np.polyval(k, xx)

lookback = 100
modelo2 = PandasRollingOLS(y=y, x=x, window=lookback)
data = data[lookback-1:]
betas = modelo2.beta

data['beta'] = betas

data['numunits'] = data.apply(lambda x : x[x_ticker] - x['beta'] * x[y_ticker], axis=1)

model = smf.OLS(y, x)
results = model.fit()

def cointegration_test(y, x):
    ols_result  = smf.OLS(y, x).fit()
    return ts.adfuller(ols_result.resid, maxlag=1)

resultsCOIN = cointegration_test(y, x)

"""
Here, we assume y is independent. Actually which one is independent variable should be test too.

The cointegration test result is 
(-2.6811245633420886, 0.07736188371900646, 1, 1498, {'1%': -3.4347228578139943, '5%': -2.863471337969528, '10%': -2.5677982210726897}, 1026.451377111122)
The return is test_statistics, pvalue, lag, #observations, critical value
"""
print('********* Cointegration test results for %s & %s *********'%(x_ticker, y_ticker))
print('cointegration t-stats: %s'%resultsCOIN[0])
print('cointegration pvalue: %s'%resultsCOIN[1])
print('cointegration used_lags: %s'%resultsCOIN[2])
print('cointegration number of observations: %s'%resultsCOIN[3])
print('cointegration critical values: %s'%resultsCOIN[4])


fig = plt.figure()
ax = fig.add_subplot(311)
data2 = data[[x_ticker, y_ticker]]
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.set_title("Share price of " + x_ticker + " versus " + y_ticker )
ax.set_xlabel(x_ticker)
ax.set_ylabel(y_ticker)
ax.plot(data2)

ax = fig.add_subplot(312)
ax.plot(data['numunits'])
ax.set_title(y_ticker + " - hedge_ratio * " + x_ticker )
ax.set_xlabel(x_ticker)
ax.set_ylabel(y_ticker)


ax = fig.add_subplot(313)
ax.plot(x, y, 'o')
ax.plot(xx, yy, 'r')
ax.set_xlabel(x_ticker)
ax.set_ylabel(y_ticker)

plt.show()
