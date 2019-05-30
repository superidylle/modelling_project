# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import config
import statsmodels.formula.api as smf
from pyfinance.ols import OLS, RollingOLS, PandasRollingOLS
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
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


# Plot

rc = {
            'lines.linewidth': 1.0,
            'axes.facecolor': '0.995',
            'figure.facecolor': '0.97',
            'font.family': 'serif',
            'font.serif': 'Ubuntu',
            'font.monospace': 'Ubuntu Mono',
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.labelweight': 'bold',
            'axes.titlesize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 10,
            'figure.titlesize': 12
        }

sns.set_context(rc)
sns.set_style("whitegrid")
sns.set_palette("deep", desat=.6)
vertical_sections = 3

fig = plt.figure(figsize=(10, vertical_sections * 3.5))
# fig.suptitle(self.title, y=0.94, weight='bold')
# fig.suptitle(self.title, weight='bold')

gs = gridspec.GridSpec(vertical_sections, 1, wspace=0.25, hspace=0.5)


def format_perc(x, pos):
    return '%.2f%%' % x


# ax = fig.add_subplot(311)

# data2 = data[[x_ticker, y_ticker]]

data.reset_index(inplace=True, drop=False)
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(by=['Date'], inplace=True)
data = data.set_index('Date')
# print(data[[x_ticker]])
# exit()


ax1 = plt.subplot(gs[0, 0])
ax1 = plt.gca()
ax1.yaxis.grid(linestyle=':')
ax1.xaxis.set_tick_params(reset=True)
ax1.xaxis.set_major_locator(mdates.YearLocator(1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.xaxis.grid(linestyle=':')
data[[x_ticker]].plot(lw=2, color='green', label=x_ticker, alpha=0.60, ax=ax1)
data[[y_ticker]].plot(lw=2, color='rosybrown', alpha=0.6, x_compat=False, ax=ax1)
ax1.set_title("Share price of " + x_ticker + " vs " + y_ticker )
ax1.set_xlabel("Year")

ax2 = plt.subplot(gs[1, 0])
ax2 = plt.gca()
ax2.yaxis.grid(linestyle=':')
ax2.xaxis.set_tick_params(reset=True)
ax2.xaxis.grid(linestyle=':')
ax2.plot(x, y, 'o', markersize=2, color='silver', label=x_ticker, alpha=0.60)
ax2.plot(xx, yy, lw=2, color='dodgerblue', label=x_ticker, alpha=0.60)
ax2.set_title("Scatter Plot of " + x_ticker + " vs " + y_ticker )
ax2.set_xlabel(x_ticker + " share price")
ax2.set_ylabel(y_ticker + " share price")


ax3 = plt.subplot(gs[2, 0])
ax3 = plt.gca()
ax3.yaxis.grid(linestyle=':')
ax3.xaxis.set_tick_params(reset=True)
ax3.xaxis.set_major_locator(mdates.YearLocator(1))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3.xaxis.grid(linestyle=':')
data['numunits'].plot(lw=2, color='coral', label=x_ticker, alpha=0.60, ax=ax3)
ax3.set_xlabel("Year")
ax3.set_ylabel(y_ticker + "- hedge ratio * " + x_ticker)

plt.show()
fig.savefig(config.output_data_path + "/CADF_test" + ".png" , dpi=150, bbox_inches='tight')
