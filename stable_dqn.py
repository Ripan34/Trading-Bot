import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import yfinance as yf
import pandas as pd
import gymnasium as gym
from gymnasium import RewardWrapper
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import DQN
import matplotlib.pyplot as plt

# symbol = "MSFT"
# df = yf.download(symbol, period="7y", interval="1d")
# df = df.dropna()[["Open", "High", "Low", "Close", "Volume"]]
# print(f"Downloaded {len(df)} rows of {symbol} data")
df = pd.read_csv('stocks_appl.csv')
df.rename(columns={"close": "Close", "high": "High", "low": "Low", "open": "Open", "volume": "Volume"}, inplace=True)

class PnLRewardWrapper(RewardWrapper):
    def __init__(self, env: StocksEnv):
        env.trade_fee_bid_percent = 0.0
        env.trade_fee_ask_percent = 0.0
        super().__init__(env)
        self.prev_equity = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_equity = self.env._total_profit
        return obs, info

    def reward(self, _: float) -> float:
        curr = self.env._total_profit
        pnl  = curr - self.prev_equity
        self.prev_equity = curr
        return pnl

WINDOW = 24  
START  = WINDOW
END    = len(df) - 168   

def make_env(frame_end):
    base = StocksEnv(df=df, window_size=WINDOW, frame_bound=(3310, frame_end))
    return PnLRewardWrapper(base)

train_env = make_env(END)

eval_env = make_env(len(df))

# model = DQN(
#     policy="MlpPolicy",
#     env=train_env,
#     verbose=1,
#     learning_rate=1e-4,
#     buffer_size=50_000,
#     learning_starts=1_000,
#     batch_size=32,
#     train_freq=4,
#     target_update_interval=1_000,
# )

# model.learn(total_timesteps=200_000)
# model.save("dqn_msft_hourly")

model = DQN.load("dqn_msft_hourly")

# 5. Run a single evaluation episode
obs, info = eval_env.reset()
terminated = truncated = False

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)

print("Final evaluation statistics:", info)

print(f"Info dict    : {info}")
profit_ratio = info['total_profit'] 

start_bankroll = 10_000  
profit_ratio   = info['total_profit']

start_value  = start_bankroll
end_value    = start_bankroll * profit_ratio

print(f"Start: ${start_value:,.2f}")
print(f"End  : ${end_value:,.2f}")
print(f"Return: {(profit_ratio-1)*100:.2f}%")

plt.cla()
eval_env.unwrapped.render_all()
plt.show()