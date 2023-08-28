import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from pylab import rcParams
rcParams['figure.figsize'] = 10, 6

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

#getting the stock prices directly
import yfinance as yf
stock_data = yf.download('^N100', start='2023-07-01', end='2023-08-15')
stock_data

# seeing the closing prices of the stocks of each data
df_close  = stock_data.Close

#plotting the close price
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Close Prices')
plt.plot(stock_data['Close'])
plt.title('Apple closing price')
plt.show()


# using the dickey fuller test to check if there is stationarity in the data ?? 
def test_stationarity(df_close):
    adf = adfuller(df_close)

    for key, value in adf[4].items():
        print(f"  {key}: {value}")

    if adf[1] < 0.05:
        print('The data is stationary')
    else:
        print('The data is not stationary')

test_stationarity(df_close)

result = seasonal_decompose(df_close, period=100 ,model='additive')

fig = plt.figure()
fig = result.plot()
fig.set_size_inches(10, 8)
plt.show()


# now using the moving averages and the standard deviations, let's eliminate the trends
from pylab import rcParams

rcParams['figure.figsize'] = 10, 6
df_log = np.log(df_close)

moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()


plt.legend(loc='best')
plt.title('Moving Average')
plt.plot(std_dev, color ="black", label = "Standard Deviation")
plt.plot(moving_avg, color="red", label = "Mean")
plt.legend()
plt.show()


train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]

plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_log, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()

model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
print(model_autoARIMA.summary())
model_autoARIMA.plot_diagnostics(figsize=(15,8))
plt.show()


def train(df_close , p , d , q):

  model = ARIMA(df_close ,order=(p, d, q))
  model = model.fit()
  return model

p = 1
q = 5
d = 2

model = train(df_close.values , p , d , q)

fc = model.forecast(321, alpha=0.05)
fc_series = pd.Series(fc, index=test_data.index)
fc_series

plt.figure(figsize=(10,5), dpi=100)
plt.plot(train_data, label='training data')
plt.plot(test_data, color = 'blue', label='Actual Stock Price')
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Apple Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()


rmse = np.sqrt(mean_squared_error(test_data, fc_series))
mae = mean_absolute_error(test_data, fc_series)

print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)