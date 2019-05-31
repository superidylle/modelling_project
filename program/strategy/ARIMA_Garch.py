# -*- coding: utf-8 -*-

import os
import sys

import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import config
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model

import matplotlib.pyplot as plt
import matplotlib as mpl

end = '2015-01-01'
start = '2007-01-01'
get_px = lambda x: yf.download(x, start=start, end=end)['Adj Close']

symbols = ['SPY','TLT','MSFT']
# raw adjusted close prices
data = pd.DataFrame({sym:get_px(sym) for sym in symbols})
# log returns
lrets = np.log(data/data.shift(1)).dropna()
