# generate_all_gifs.py

import os
import sys
import time
import pygame
import imageio
import numpy as np

from stable_baselines3 import PPO, DQN, A2C

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Environment.custom_env import FishFeedingEnv
from Environment.rendering import FishFeedingRenderer

def record_agent(env, model=None, gif_name="agent.gif", max_steps=100):
    renderer = FishFeedingRenderer(env)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs

    frames = []

    for step in range(max_steps):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        renderer.render(step=step, reward=reward)
        surface = pygame.display.get_surface()
        frame_array = pygame.surfarray.array3d(surface)
        frame_array = frame_array.transpose([1, 0, 2])
        frames.append(frame_array)

        time.sleep(0.1)
        if terminated or truncated:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs

    renderer.close()
    imageio.mimsave(gif_name, frames, duration=0.7)
    print(f"✅ Saved: {gif_name}")

def load_model(path, model_type):
    if not os.path.exists(path):
        print(f"❌ Model not found at: {path}")
        return None

    if model_type == "ppo":
        return PPO.load(path)
    elif model_type == "dqn":
        return DQN.load(path)
    elif model_type == "a2c":
        return A2C.load(path)
    else:
        raise ValueError("Unsupported model type.")

if __name__ == "__main__":
    os.makedirs("gifs", exist_ok=True)

    # 1. Random Agent
    print("🎬 Generating Random Agent GIF...")
    env = FishFeedingEnv(render_mode=None)
    record_agent(env, model=None, gif_name="gifs/fish_agent_random.gif")

    # 2. DQN Agent
    print("🎬 Generating DQN Agent GIF...")
    model = load_model("./models/dqn/best_model/best_model.zip", "dqn")
    if model:
        env = FishFeedingEnv(render_mode=None)
        record_agent(env, model, gif_name="gifs/fish_agent_dqn.gif")

    # 3. PPO Agent
    print("🎬 Generating PPO Agent GIF...")
    model = load_model("./models/ppo/best_model.zip", "ppo")
    if model:
        env = FishFeedingEnv(render_mode=None)
        record_agent(env, model, gif_name="gifs/fish_agent_ppo.gif")

    # 4. REINFORCE (simulated with A2C)
    print("🎬 Generating REINFORCE Agent GIF...")
    model = load_model("./models/pg/reinforce_model2.zip", "a2c")
    if model:
        env = FishFeedingEnv(render_mode=None)
        record_agent(env, model, gif_name="gifs/fish_agent_reinforce.gif")

    # 5. A2C Agent
    print("🎬 Generating A2C Agent GIF...")
    model = load_model("./models/a2c/best_model.zip", "a2c")
    if model:
        env = FishFeedingEnv(render_mode=None)
        record_agent(env, model, gif_name="gifs/fish_agent_a2c.gif")

    # 6. Best Overall (currently set to PPO)
    print("🎬 Generating Best Overall Agent GIF (PPO)...")
    model = load_model("./models/ppo/best_model.zip", "ppo")  # change if needed
    if model:
        env = FishFeedingEnv(render_mode=None)
        record_agent(env, model, gif_name="gifs/fish_agent_best.gif")

    print("🎉 All GIFs generated in the 'gifs/' folder!")
