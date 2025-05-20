import torch
import pandas as pd
import gymnasium as gym
import gym_trading_env
from nn import QNetwork

df = pd.read_csv("stocks_appl.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df.set_index("Datetime", inplace=True)

def zscore(obs):
    return (obs - obs.mean(axis=0)) / (obs.std(axis=0) + 1e-8)

env_raw = gym.make(
    "TradingEnv",
    df=df,
    positions=[0, 0.5, 1],
    windows=10,
    initial_position=0,
    portfolio_initial_value=10000,
)

env = gym.wrappers.TransformObservation(
    env_raw,
    zscore,
    observation_space=env_raw.observation_space  # <- satisfy the wrapper
)

state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
action_size = env.action_space.n

model = QNetwork(state_size, action_size)
model.load_state_dict(torch.load("trading_q_network.pth"))
model.eval()

obs, info = env.reset()
done = False
total_reward = 0

while not done:
    with torch.no_grad():
        state_tensor = torch.FloatTensor(obs).view(1, -1)
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

    env.unwrapped.save_for_render("render_eval")

print(f"Evaluation finished — Total reward: {total_reward:.2f}")
env.close()
