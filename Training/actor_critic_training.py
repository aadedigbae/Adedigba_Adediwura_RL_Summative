import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from Environment.custom_env import FishFeedingEnv
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

# Wrap environments
env = Monitor(FishFeedingEnv())
eval_env = Monitor(FishFeedingEnv())

# Callback to stop training early
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/a2c/",
    log_path="./logs/a2c/",
    eval_freq=10000,
    n_eval_episodes=10,
    deterministic=True,
    callback_after_eval=StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        verbose=1
    )
)

# Create A2C model
model = A2C(
    policy="MlpPolicy",
    env=env,
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    gae_lambda=1.0,
    ent_coef=0.01,
    vf_coef=0.25,
    max_grad_norm=0.5,
    use_rms_prop=True,
    verbose=1
)

# Train
model.learn(total_timesteps=300_000, callback=eval_callback)

# Final Evaluation
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"✅ A2C Evaluation - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
