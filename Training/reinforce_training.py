import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from Environment.custom_env import FishFeedingEnv
from stable_baselines3 import A2C  # We'll simulate REINFORCE using A2C with tiny entropy bonus
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
import os

# Create envs
env = Monitor(FishFeedingEnv())
eval_env = Monitor(FishFeedingEnv())

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./models/pg_checkpoints/",
    name_prefix="reinforce_checkpoint"
)

model = A2C(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    ent_coef=0.02,
    vf_coef=0.25,
    normalize_advantage=True,
    gamma=0.99,
    n_steps=1024,
    gae_lambda=0.92,
    tensorboard_log="./logs/reinforce/",
    verbose=1
)

# Train longer
model.learn(total_timesteps=200_000, callback=checkpoint_callback)

# Save model
os.makedirs("models/pg", exist_ok=True)
model.save("models/pg/reinforce_model2")

# Evaluate
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"✅ REINFORCE Evaluation - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
