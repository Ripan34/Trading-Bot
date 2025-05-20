import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import yfinance as yf
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import RewardWrapper
import ta
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import matplotlib.pyplot as plt

df = pd.read_csv('stocks_appl.csv')
df.rename(columns={"close": "Close", "high": "High", "low": "Low", "open": "Open", "volume": "Volume"}, inplace=True)

df["rsi_14"]    = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
df["macd"]      = ta.trend.MACD(df["Close"]).macd_diff()
df["return_1h"] = df["Close"].pct_change().fillna(0)

df.dropna(inplace=True)
print(f"After feature engineering: {df.shape[0]} rows, {df.shape[1]} columns")

class MyStocksEnv(StocksEnv):
    trade_fee_bid_percent = 0.0
    trade_fee_ask_percent = 0.0

    def _process_data(self):
        start, end = self.frame_bound
        prices = self.df["Close"].to_numpy()[start:end]
        diff   = np.insert(np.diff(prices), 0, 0)
        feats  = self.df[["rsi_14", "macd", "return_1h"]].to_numpy()[start:end]
        signal_features = np.column_stack((prices, diff, feats))
        return prices.astype(np.float32), signal_features.astype(np.float32)

class PctReturnRewardWrapper(RewardWrapper):
    def __init__(self, env: StocksEnv):
        super().__init__(env)
        self.prev_equity = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_equity = self.env._total_profit
        return obs, info

    def reward(self, _: float) -> float:
        curr = self.env._total_profit
        pct  = (curr - self.prev_equity) / (self.prev_equity + 1e-8)
        self.prev_equity = curr
        return pct

WINDOW = 24
START  = WINDOW
END    = len(df) - 168 

def make_env(frame_end):
    base = MyStocksEnv(
        df=df,
        window_size=WINDOW,
        frame_bound=(3200, frame_end)
    )
    return PctReturnRewardWrapper(base)

train_env = DummyVecEnv([lambda: make_env(END)])
train_env = VecMonitor(train_env)

eval_env = make_env(len(df))


model = A2C.load("a2c_msft_hourly_custom")

obs, info = eval_env.reset()
terminated = truncated = False
while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)

print("Final evaluation stats:", info)
plt.cla()
eval_env.unwrapped.render_all() 
plt.show()