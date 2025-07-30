# play_render.py

import time
from custom_env import FishFeedingEnv
from rendering import FishFeedingRenderer

# Initialize environment and renderer
env = FishFeedingEnv()
renderer = FishFeedingRenderer(env)

obs = env.reset()
if isinstance(obs, tuple):  # gymnasium returns (obs, info)
    obs, _ = obs

done = False

for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(f"[Step {step}] Action: {action}, Reward: {reward:.2f}, Info: {info}")

    renderer.render()
    time.sleep(0.3)

    if done:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs

renderer.close()
