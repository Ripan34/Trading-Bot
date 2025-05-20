import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import yfinance as yf
import pandas as pd
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import matplotlib.pyplot as plt

df = pd.read_csv('stocks_appl.csv')
df.rename(columns={"close": "Close", "high": "High", "low": "Low",
                   "open": "Open", "volume": "Volume"}, inplace=True)
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df[feature_cols] = df[feature_cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))

class HourlyStocksEnv(StocksEnv):
    _process_data = StocksEnv._process_data
    _calculate_reward = StocksEnv._calculate_reward

window_size = 24 
start = window_size
end = len(df) - 168  

def make_train_env():
    return HourlyStocksEnv(df=df, window_size=window_size, frame_bound=(start, end))

train_env = DummyVecEnv([make_train_env])

eval_env = HourlyStocksEnv(df=df, window_size=window_size, frame_bound=(1400, 1600))

model = PPO.load("ppo_hourly_msft_final")

obs, info = eval_env.reset()
terminated = truncated = False

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)

print("Final evaluation statistics:", info)

start_bankroll = 10_000
profit_ratio   = info['total_profit']
end_value      = start_bankroll * profit_ratio

print(f"Start: ${start_bankroll:,.2f}")
print(f"End  : ${end_value:,.2f}")
print(f"Return: {(profit_ratio-1)*100:.2f}%")

plt.cla()
eval_env.unwrapped.render_all()
plt.show()
