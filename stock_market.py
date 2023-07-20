import pandas as pd
import numpy as np
import gym
from gym import spaces
import talib as ta
import torch
import torch.nn as nn
import torch.optim as optim

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

        # Extract relevant data from stock_data (assuming stock_data is a Pandas DataFrame)
        self.stock_prices = stock_data['Adj Close'].values
        self.high_prices = stock_data['High'].values
        self.low_prices = stock_data['Low'].values
        self.num_stocks = len(stock_data.columns) - 1  # Number of stocks (excluding 'Date' column)
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

        # Initialize the Actor-Critic model
        self.input_dim = self.num_stocks * 10 + 1
        self.output_dim = self.num_stocks
        self.model = ActorCriticModel(self.input_dim, self.output_dim)

        # Set the optimizer for the model
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Initialize the state
        self.shares = np.zeros(self.num_stocks)
        self.portfolio_value = self.balance

        # Technical indicators parameters
        self.window = 14  # Window for RSI and CCI calculation

    def reset(self):
        # Reset the environment to the initial state
        self.balance = 10000.0
        self.shares = np.zeros(self.num_stocks)
        self.portfolio_value = self.balance
        return self._get_observation()

    def step(self, action_probs):
        # Execute the action and calculate the new state, reward, and other information
        actions = np.random.choice(np.arange(-self.max_shares, self.max_shares+1), size=self.num_stocks, p=action_probs)
        actions = actions / self.max_shares  # Normalize the actions to [-1, 1]

        # Update stock prices based on the actions taken
        self.stock_prices *= (1 + actions)

        # Calculate the portfolio value after the actions
        portfolio_value = np.sum(self.stock_prices * self.shares) + self.balance

        # Calculate the reward as the change in portfolio value
        reward = portfolio_value - self.portfolio_value
        self.portfolio_value = portfolio_value

        # Update balance and shares based on the actions taken
        self.balance -= np.sum(self.stock_prices * actions * self.shares)
        self.shares += actions

        # Calculate the new observation/state
        new_observation = self._get_observation()

        # Check if the episode is done (e.g., based on a fixed time step or other criteria)
        done = False

        return new_observation, reward, done

    def _get_observation(self):
        # Calculate MACD
        macd, signal_line, _ = ta.MACD(self.stock_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        # Normalize MACD and signal line to [0, 1]
        macd = (macd - np.min(macd)) / (np.max(macd) - np.min(macd))
        signal_line = (signal_line - np.min(signal_line)) / (np.max(signal_line) - np.min(signal_line))

        # Calculate RSI
        rsi = ta.RSI(self.stock_prices, timeperiod=self.window)
        # Normalize RSI to [0, 1]
        rsi = rsi / 100.0

        # Calculate CCI
        cci = ta.CCI(self.high_prices, self.low_prices, self.stock_prices, timeperiod=self.window)
        # Normalize CCI to [-1, 1]
        cci = (cci - np.min(cci)) / (np.max(cci) - np.min(cci)) * 2 - 1

        # Calculate ADX
        adx = ta.ADX(self.high_prices, self.low_prices, self.stock_prices, timeperiod=self.window)
        # Normalize ADX to [-1, 1]
        adx = (adx - np.min(adx)) / (np.max(adx) - np.min(adx)) * 2 - 1

        # Create the observation vector
        observation = np.hstack([self.shares / self.max_shares,  # Normalized shares
                                 self.stock_prices / np.max(self.stock_prices),  # Normalized stock prices
                                 rsi,
                                 cci,
                                 adx,
                                 self.balance / 20000.0])  # Normalized balance

        return observation

# Load the stock data from the CSV file
data = pd.read_csv('^NSEI.csv', parse_dates=['Date'])

# Create the environment
max_shares = 1000  # You can adjust this based on your requirements
env = MultiStockTradingEnv(data, max_shares)
