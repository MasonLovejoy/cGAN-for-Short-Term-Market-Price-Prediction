# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:10:58 2024

@author: Mason

______________________________________________________

Preprocessing Ideas:
    
    (1) Fourier Discritization of Data
    (2) Differential Processing
    
    Im gonna train before I add Time Series Decompositions because I am probably
    gonna need to get a years worth of data to do that.


CODE FORE ARIMA ANALYSIS:
    --------
    py.plot(data['Close'])
    py.plot(test['Close'])
    py.show()
    
    series_index = pd.to_datetime(data['Date'])
    data_series = data['Close']
    data_series.index = series_index
    data_series.index = data_series.index.to_period('D')
    
    model = ARIMA(data_series, order=(5,1,0))
    model_fit = model.fit()
    model_test = model.prepare_data()
    --------
"""

from scipy.fft import fft, ifft
import numpy as np
import pandas as pd
import matplotlib.pyplot as py
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
import torch
from scipy.stats import zscore

nvda_data = pd.read_csv('data\\NVDA.csv')

def append_columns(data: pd.DataFrame, append_data: pd.DataFrame, feature: str) -> pd.DataFrame:
    for column in append_data:
        name = feature + ' ' + column
        data.insert(loc=0, column=name, value=append_data[column])
    return data

def bollinger_bands(series: pd.Series, length: int = 20, *, num_stds: tuple[float, ...] = (2, 0, -2), prefix: str = '') -> pd.DataFrame:
    rolling = series.rolling(length)
    bband0 = rolling.mean()
    bband_std = rolling.std(ddof=0)
    return pd.DataFrame({f'{prefix}{num_std}': (bband0 + (bband_std * num_std)) for num_std in num_stds})
        
def preprocessing(data: pd.DataFrame, data_length: int) -> pd.DataFrame:
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop('Volume')
    
    volumes = data['Volume']
    volume_change = data['Volume'].pct_change()
    dates = data['Date']
    
    data = data[numeric_cols]
    
    # in the future may want to play with some for the rolling windows
    roll_avg_7d = data.rolling(window=7).mean()
    roll_avg_21d = data.rolling(window=21).mean()
    
    #close_fft = fft(data['Close'].values)
    
    # setting these to days of four in order to help model
    # because the data will be sent into the cgan at 3-day 
    # intervals
    exp_avg = data.ewm(span=4).mean()
    momentum = data.pct_change(4)
    bbl_close = bollinger_bands(data['Close'], length = 4, num_stds=(2,0,-2))
    bbl_adj = bollinger_bands(data['Adj Close'], length = 4, num_stds=(2,0,-2))
    
    
    ema_12 = data.ewm(span=12).mean()
    ema_26 = data.ewm(span=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9).mean()
    
    data = append_columns(data, roll_avg_7d, '7 Day Rolling Avg')
    data = append_columns(data, roll_avg_21d, '21 Day Rolling Avg')
    data = append_columns(data, exp_avg, 'Exponential Rolling Avg')
    data = append_columns(data, bbl_close, 'Close Bands')
    data = append_columns(data, bbl_adj, 'Adj Close Bands')
    data = append_columns(data, macd, 'MACD')
    data = append_columns(data, macd_signal, 'MACD Signal')
    data = append_columns(data, momentum, 'Momentum')

    #data = data.apply(np.log)
    
    #data.insert(loc=0, column='Volume', value=volumes)
    
    data = data.tail(data_length+1)
    data = data.apply(zscore)
    
    data.insert(loc=0, column='Volume Change', value=volume_change)
    #data.insert(loc=0, column='Dates', value=dates)
    return data
    
nvda_data = preprocessing(nvda_data, 1000)
nvda_data = nvda_data[:31]
test_data = nvda_data.tail(1)

# dataset parameters
window_len = 4
window_amt = int(len(nvda_data[:])-window_len)
features = len(nvda_data.columns)

torch_tensors  = torch.zeros((window_amt, window_len, features))

for i in range(0, window_amt):
    torch_tensors[i,:,:] = torch.tensor(nvda_data[i:i+window_len].values)




