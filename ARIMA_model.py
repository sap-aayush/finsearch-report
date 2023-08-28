import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import gym

nifty100_data = yf.download('^N100', start='2023-07-01', end='2023-08-15')

nifty100_data['Returns'] = nifty100_data['Adj Close'].pct_change().fillna(0)

class TradingEnvironment(gym.Env):
    def __init__(self, data):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        self.current_step = 0
        self.initial_cash = 100000  # Initial investment
        self.cash = self.initial_cash
        self.shares_held = 0
        self.current_price = 0
        self.portfolio_value = self.cash
    
    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.portfolio_value = self.cash
        return self._get_observation()
    
    def _get_observation(self):
        return np.array([self.cash / self.initial_cash, self.shares_held])
    
    def step(self, action):
        self.current_price = self.data['Adj Close'][self.current_step]
        
        if action == 0:  # Buy
            self.shares_held += self.cash / self.current_price
            self.cash = 0
        elif action == 1:  # Sell
            self.cash += self.shares_held * self.current_price
            self.shares_held = 0
        
        self.portfolio_value = self.cash + self.shares_held * self.current_price
        
        self.current_step += 1
        done = self.current_step >= len(self.data)
        
        return self._get_observation(), self.portfolio_value, done, {}

