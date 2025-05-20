import gymnasium as gym
import gym_anytrading
import numpy as np
import random
import torch
from torch import nn, optim
import pandas as pd
import ta
from nn import QNetwork, ReplayBuffer
from gym_anytrading.envs import StocksEnv
import gymnasium as gym
from gymnasium import RewardWrapper
import matplotlib.pyplot as plt

ENV_ID        = "stocks-v0"
WINDOW_SIZE   = 20
FRAME_BOUND   = (50, 3200)
NUM_EPISODES  = 500
BATCH_SIZE    = 32
GAMMA         = 0.99
EPS_START     = 1.0
EPS_MIN       = 0.01
EPS_DECAY     = 0.998
TARGET_SYNC   = 15
LR            = 1e-4
REPLAY_CAP    = 100_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('stocks_appl.csv')
df.rename(columns={"close": "Close", "high": "High", "low": "Low", "open": "Open", "volume": "Volume"}, inplace=True)

df['SMA_20']  = ta.trend.sma_indicator(df['Close'], window=20)
df['EMA_20']  = ta.trend.ema_indicator(df['Close'], window=20)
df['MACD']    = ta.trend.macd_diff(df['Close'])
df['BB_bbm']  = ta.volatility.bollinger_mavg(df['Close'], window=20)


feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 'MACD', 'BB_bbm']
df[feature_cols] = df[feature_cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))

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
    base = StocksEnv(df=df, window_size=WINDOW, frame_bound=(2900, 3000))
    return PnLRewardWrapper(base)

eval_env = make_env(len(df))

env = make_env(len(df))
obs, info = env.reset()

state_size  = int(np.prod(obs.shape))
action_size = env.action_space.n
print(f"state_size={state_size}, action_size={action_size}")

class TradingAgentQN:
    def __init__(self):
        self.q_network      = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.memory = ReplayBuffer(REPLAY_CAP)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.epsilon = EPS_START

    def act(self, state):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        state_t = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = self.q_network(state_t)
        return torch.argmax(q_vals).item()

    def remember(self, s, a, r, s2, done):
        self.memory.push(s, a, r, s2, done)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states      = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions     = torch.LongTensor(actions).to(device)
        rewards     = torch.FloatTensor(rewards).to(device)
        dones       = torch.FloatTensor(dones).to(device)

        q_values  = self.q_network(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q     = self.target_network(next_states)
            max_next_q = torch.max(next_q, dim=1)[0]
            target_q   = rewards + GAMMA * max_next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

# agent = TradingAgentQN()

# for ep in range(1, NUM_EPISODES + 1):
#     obs, info = env.reset()
#     total_reward = 0
#     done = False

#     while not done:
#         action = agent.act(obs)
#         next_obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated

#         agent.remember(obs, action, reward, next_obs, done)
#         agent.learn()

#         obs = next_obs
#         total_reward += reward

#     agent.decay_epsilon()
#     if ep % TARGET_SYNC == 0:
#         agent.update_target()

#     print(f"Episode {ep:3d} | Reward {total_reward:8.2f} | ε={agent.epsilon:5.3f}")

# agent.save(path="hourly_q_network.pth")
# env.close()
# print("Training complete")

MODEL_PATH        = "hourly_q_network.pth"
model = QNetwork(state_size, action_size).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

obs, info = eval_env.reset()
terminated = truncated = False

total_reward = 0.0
step = 0
while True:
    state_t = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_t)
        action   = torch.argmax(q_values).item()          # greedy
    obs, reward, terminated, truncated, info = eval_env.step(action)

    total_reward += reward
    step += 1
    #env.render()
    if terminated or truncated:
        break

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
