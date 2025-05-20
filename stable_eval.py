import gymnasium as gym
import gym_anytrading
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Load and preprocess data
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df = pd.read_csv('stocks_appl.csv', parse_dates=['Datetime'], index_col='Datetime')
df.rename(columns={"close": "Close", "high": "High", "low": "Low", "open": "Open", "volume": "Volume"}, inplace=True)

df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
df[feature_cols] = (df[feature_cols] - df[feature_cols].min()) / (df[feature_cols].max() - df[feature_cols].min())

window_size = 30
frame_bound = (3000, 3100) 

env = gym.make('stocks-v0', df=df, window_size=window_size, frame_bound=frame_bound)
vec_env = DummyVecEnv([lambda: env])

model = DQN.load("dqn_aapl_trading", env=vec_env)

obs = vec_env.reset()
done = False
total_reward = 0
step = 0
equity_curve = []

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    total_reward += reward[0]
    step += 1
    env.render()
    equity_curve.append(env.unwrapped._total_profit)

    done = done[0]

print(f"\nTest finished in {step} steps")
print(f"Total reward : {total_reward:.2f}")
print(f"Info dict    : {info[0]}")

profit_ratio = info[0]['total_profit']
start_bankroll = 10_000
end_value = start_bankroll * profit_ratio

print(f"\nStart: ${start_bankroll:,.2f}")
print(f"End  : ${end_value:,.2f}")
print(f"Return: {(profit_ratio - 1) * 100:.2f}%")

plt.figure(figsize=(12, 5))
plt.plot([start_bankroll * p for p in equity_curve], label="Equity Curve")
plt.title("Agent Equity Curve During Evaluation")
plt.xlabel("Step")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 6))
#env.unwrapped.render_all()
plt.show()