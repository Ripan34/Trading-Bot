import yfinance as yf
import pandas as pd
import gymnasium as gym
import gym_trading_env
# from fetch_data import get_data
from trading_agent_qn import TradingAgentQN

#data = get_data(ticker = "AAPL", interval = "1h", period="1y")
df = pd.read_csv('stocks_appl.csv')
df["Datetime"] = pd.to_datetime(df["Datetime"])
df.set_index("Datetime", inplace=True)

feature_cols = ['open', 'high', 'low', 'close', 'volume']
#df[feature_cols] = (df[feature_cols] - df[feature_cols].min()) / (df[feature_cols].max() - df[feature_cols].min())

# Rename to match expected by gym-trading
# df.rename(columns={"close": "Close", "high": "High", "low": "Low", "open": "Open", "volume": "Volume"}, inplace=True)
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

# === Setup Agent ===
obs, info = env.reset()
state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
action_size = env.action_space.n

agent = TradingAgentQN(
    env=env,
    action_size=action_size,
    state_size=state_size,
    epsilon=1.0,
    batch_size=32,
    gama=0.99,
    min_epsilon=0.01,
    epsilon_decay=0.997
)

num_episodes = 300
target_update_frequency = 15

for episode in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.push_buffer(obs, action, reward, next_obs, done)
        agent.update(episode, target_update_frequency)

        obs = next_obs
        total_reward += reward
        env.unwrapped.save_for_render(dir = "render_logs")

    agent.decay_epsilon()
    print(f"Episode {episode+1} — Total Reward: {total_reward:.2f} — Epsilon: {agent.epsilon:.3f}")

agent.save_model()
env.close()

# if not data.empty:
#     data.to_csv('stocks_appl.csv', index=False)

#     env = gym.make("TradingEnv",
#         name= "APPL",
#         df = data,
#         positions = [0, 0.3, 0.5, 1],
#     )

#     print("Environment created successfully with action space:", env.action_space)
#     print("Observation space:", env.observation_space)

#     observation, info = env.reset()
#     done = False
#     total_reward = 0

#     while not done:
#         action = env.action_space.sample()
#         observation, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         total_reward += reward
#         env.save_for_render(dir = "render_logs")
#         #print(f"Episode finished with total reward: {total_reward}")

#     env.close()

# else:
#     print("Could not retrieve data from yfinance.")