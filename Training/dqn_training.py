import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from Environment.custom_env import FishFeedingEnv
import os


env = FishFeedingEnv()
env = Monitor(env)

eval_env = FishFeedingEnv()
eval_env = Monitor(eval_env)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/dqn/best_model",
    log_path="./models/dqn/logs",
    eval_freq=5000,
    deterministic=True,
    render=False
)

model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=250,
    verbose=1
)

model.learn(total_timesteps=100000, callback=eval_callback)

# Save the model
os.makedirs("models/dqn", exist_ok=True)
model.save("models/dqn/fish_dqn_model")

# Evaluate
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"✅ DQN Evaluation - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
