import numpy as np
import gymnasium as gym
import gym_anytrading
import torch
import matplotlib.pyplot as plt
from nn import QNetwork         
import pandas as pd
import ta
from gym_anytrading.envs import StocksEnv

ENV_ID            = "stocks-v0" 
WINDOW_SIZE       = 20
TEST_FRAME_BOUND  = (3300, 3450)  
MODEL_PATH        = "hourly_q_network.pth" 
#SEED              = 2025

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('stocks_appl.csv')
df.rename(columns={"close": "Close", "high": "High", "low": "Low", "open": "Open", "volume": "Volume"}, inplace=True)

df['SMA_20']  = ta.trend.sma_indicator(df['Close'], window=20)
df['EMA_20']  = ta.trend.ema_indicator(df['Close'], window=20)
df['MACD']    = ta.trend.macd_diff(df['Close'])
df['BB_bbm']  = ta.volatility.bollinger_mavg(df['Close'], window=20)

feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 'MACD', 'BB_bbm']
df[feature_cols] = df[feature_cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))

class CustomStocksEnv(StocksEnv):
    def __init__(self, df, window_size=20):
        self.df = df
        self.window_size = window_size
        self.frame_bound = TEST_FRAME_BOUND
        super().__init__(df=self.df, window_size=self.window_size, frame_bound=TEST_FRAME_BOUND)

    def _process_data(self):
        prices = self.df['Close'].values
        signal_features = self.df[feature_cols].values
        return prices, signal_features

env = CustomStocksEnv(df=df, window_size=WINDOW_SIZE)

obs, info = env.reset()

state_size  = int(np.prod(obs.shape)) 
action_size = env.action_space.n    

q_net = QNetwork(state_size, action_size).to(device)
q_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
q_net.eval()

total_reward = 0.0
step = 0
while True:
    state_t = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = q_net(state_t)
        action   = torch.argmax(q_values).item()          # greedy
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    step += 1
    #env.render()
    if terminated or truncated:
        break

print(f"\n===== Test finished in {step} steps =====")
print(f"Total reward : {total_reward:.2f}")
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
env.unwrapped.render_all() 
plt.show()

env.close()
