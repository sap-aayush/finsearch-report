import pandas as pd
import numpy as np
import gym
from gym import spaces
import talib as ta
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class ActorCriticModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCriticModel, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

class MultiStockTradingEnv(gym.Env):
    def __init__(self, stock_data, max_shares):
        super(MultiStockTradingEnv, self).__init__()

        self.stock_prices = stock_data['Adj Close'].values
        self.high_prices = stock_data['High'].values
        self.low_prices = stock_data['Low'].values
        self.num_stocks = len(stock_data.columns) - 1  # Number of stocks
        self.balance = 10000.0  # Initial balance
        self.max_shares = max_shares

        # Action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_stocks,), dtype=np.float32)

        # Observation space (state space)
        obs_space_low = np.zeros(self.num_stocks * 10 + 1)  # 4 technical indicators * num_stocks + balance
        obs_space_high = np.hstack([np.ones(self.num_stocks) * max_shares,  # Maximum shares
                                    np.ones(self.num_stocks) * np.max(self.stock_prices),  # Maximum price
                                    np.ones(self.num_stocks) * 100,  # Maximum RSI value (normalized to 100)
                                    np.ones(self.num_stocks) * 200,  # Maximum CCI value (normalized to 200)
                                    np.ones(self.num_stocks) * 100,  # Maximum ADX value (normalized to 100)
                                    np.ones(self.num_stocks) * 20000])  # Maximum balance

        self.observation_space = spaces.Box(low=obs_space_low, high=obs_space_high, dtype=np.float32)

        # Actor-Critic model
        self.input_dim = self.num_stocks * 10 + 1
        self.output_dim = self.num_stocks
        self.model = ActorCriticModel(self.input_dim, self.output_dim)

        # Setting the optimizer for the model
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Initializing the state
        self.shares = np.zeros(self.num_stocks)
        self.portfolio_value = self.balance

        # Technical indicators parameters
        self.window = 14  # Window for RSI and CCI calculation

    def reset(self):
        self.balance = 10000.0
        self.shares = np.zeros(self.num_stocks)
        self.portfolio_value = self.balance
        return self._get_observation()

    def step(self, action_probs):
        actions = np.random.choice(np.arange(-self.max_shares, self.max_shares+1), size=self.num_stocks, p=action_probs)
        actions = actions / self.max_shares  # Normalizing the actions to [-1, 1]

        # Updating stock prices based on the actions
        self.stock_prices *= (1 + actions)

        # Calculating the portfolio value after the actions
        portfolio_value = np.sum(self.stock_prices * self.shares) + self.balance

        # Calculating the reward 
        reward = portfolio_value - self.portfolio_value
        self.portfolio_value = portfolio_value

        # Updating balance and shares
        self.balance -= np.sum(self.stock_prices * actions * self.shares)
        self.shares += actions

        # new observation/state
        new_observation = self._get_observation()

        # Checking if the episode is done
        done = False

        return new_observation, reward, done

    def _get_observation(self):
        # MACD
        macd, signal_line, _ = ta.MACD(self.stock_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        # Normalizing MACD and signal line to [0, 1]
        macd = (macd - np.min(macd)) / (np.max(macd) - np.min(macd))
        signal_line = (signal_line - np.min(signal_line)) / (np.max(signal_line) - np.min(signal_line))

        #  RSI
        rsi = ta.RSI(self.stock_prices, timeperiod=self.window)
        # Normalizing RSI to [0, 1]
        rsi = rsi / 100.0

        # CCI
        cci = ta.CCI(self.high_prices, self.low_prices, self.stock_prices, timeperiod=self.window)
        # Normalizing CCI to [-1, 1]
        cci = (cci - np.min(cci)) / (np.max(cci) - np.min(cci)) * 2 - 1

        # ADX
        adx = ta.ADX(self.high_prices, self.low_prices, self.stock_prices, timeperiod=self.window)
        # Normalizing ADX to [-1, 1]
        adx = (adx - np.min(adx)) / (np.max(adx) - np.min(adx)) * 2 - 1

        # observation vector
        observation = np.hstack([self.shares / self.max_shares,  # Normalized shares
                                 self.stock_prices / np.max(self.stock_prices),  # Normalized stock prices
                                 rsi,
                                 cci,
                                 adx,
                                 self.balance / 20000.0])  # Normalized balance

        return observation

# stock data from the CSV file
data = pd.read_csv('^NSEI.csv', parse_dates=['Date'])

# environment
max_shares = 1000 # taking the 100 shares for now!
env = MultiStockTradingEnv(data, max_shares)
