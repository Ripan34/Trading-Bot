import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import gymnasium as gym
from gymnasium import RewardWrapper
from gym_anytrading.envs import StocksEnv
import torch as th
import torch.nn.functional as F
from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import MlpPolicy
import matplotlib.pyplot as plt
import torch as th
import torch.nn.functional as F
from stable_baselines3.dqn import DQN

df = pd.read_csv('stocks_appl.csv')
df.rename(columns={"close": "Close", "high": "High", "low": "Low",
                   "open": "Open", "volume": "Volume"}, inplace=True)
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
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

WINDOW = 30
END    = len(df) - 168

def make_env(frame_end):
    base = StocksEnv(df=df, window_size=WINDOW, frame_bound=(1400, 1600))
    return PnLRewardWrapper(base)

train_env = make_env(END)
eval_env  = make_env(len(df))

import torch as th
import torch.nn.functional as F

class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 32) -> None:
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size,
                                                    env=self._vec_normalize_env)

            next_q_values = self.q_net(replay_data.next_observations)
            next_actions  = next_q_values.argmax(dim=1)

            next_q_target = self.q_net_target(replay_data.next_observations)
            batch_idx     = th.arange(len(next_actions))
            selected_q_t  = next_q_target[batch_idx, next_actions]

            rewards = replay_data.rewards.flatten()
            dones   = replay_data.dones.flatten()
            target_q = rewards + (1 - dones) * self.gamma * selected_q_t

            current_q = self.q_net(replay_data.observations)
            actions   = replay_data.actions.flatten().long()
            current_q = current_q[batch_idx, actions]

            loss = F.mse_loss(current_q, target_q)
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

            self._n_updates += 1

            if self._n_updates % self.target_update_interval == 0:
                self.q_net_target.load_state_dict(self.q_net.state_dict())

# model = DoubleDQN(
#     policy="MlpPolicy",
#     env=train_env,
#     learning_rate=1e-4,
#     buffer_size=50_000,
#     learning_starts=1_000,
#     batch_size=32,
#     train_freq=4,
#     target_update_interval=1_000,
#     verbose=1,
# )

# model.learn(total_timesteps=200_000)
# model.save("double_dqn_msft_hourly")

model = DoubleDQN.load("double_dqn_msft_hourly", env=eval_env)

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
