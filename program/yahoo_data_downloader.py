# -*- coding: utf-8 -*-

import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import config

def data_get_from_yahoo(symbol,start, end):

    data = yf.download(tickers=symbol,interval = "1d", start=start, end=end, treads=True)
    data.to_csv(config.input_data_path + '/' + symbol + '.csv')

# symbol = ["MSFT", "AAPL", "HON", "XOM", "WMT", "PG","AMZN", "GOOG", "JPM", "MA",
#           "VZ", "CSCO","UNH", "T", "PFE", "CVX", "HD", "MRK", "KO", "WFC", "BA",
#           "INTC", "DIS", "CMCSA", "PEP", "ORCL", "NFLX", "MCD", "C", "LLY", "PM",
#           "UNP", "MDT", "ABBV", "CRM", "IBM", "ACN", "AMGN", "UTX", "COST", "TMO",
#           "LIN", "TXN", "AXP", "NEE", "LMT", "MMM","SBUX"]
# start = "2007-08-01"
# end = "2019-05-27"
#
# for code in symbol:
#     data_get_from_yahoo(code, start, end)












