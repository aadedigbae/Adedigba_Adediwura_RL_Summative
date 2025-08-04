# record_agent_policy.py

import sys
import os
import time
import pygame
import imageio
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Environment.custom_env import FishFeedingEnv
from Environment.rendering import FishFeedingRenderer
from stable_baselines3 import PPO, DQN, A2C

def record_policy(model_path, model_type, output_gif="agent.gif", max_steps=100):
    # Load model
    if model_type == "ppo":
        model = PPO.load(model_path)
    elif model_type == "dqn":
        model = DQN.load(model_path)
    elif model_type == "a2c":
        model = A2C.load(model_path)
    else:
        raise ValueError("Unsupported model type. Use: ppo, dqn, a2c")

    # Init environment and renderer
    env = FishFeedingEnv(render_mode=None)
    renderer = FishFeedingRenderer(env)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs

    frames = []
    done = False

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        renderer.render()

        surface = pygame.display.get_surface()
        frame_array = pygame.surfarray.array3d(surface)
        frame_array = frame_array.transpose([1, 0, 2])
        frames.append(frame_array)

        time.sleep(0.1)
        if done:
            break

    renderer.close()
    imageio.mimsave(output_gif, frames, duration=0.2)
    print(f"🎥 Saved {output_gif}")

# Example usage
if __name__ == "__main__":
    record_policy(
        model_path="./models/dqn/best_model/best_model.zip",
        model_type="dqn",
        output_gif="fish_agent_dqn.gif"
    )
