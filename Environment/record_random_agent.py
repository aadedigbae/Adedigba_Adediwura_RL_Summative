import time
import pygame
import imageio
from custom_env import FishFeedingEnv
from rendering import FishFeedingRenderer

env = FishFeedingEnv()
renderer = FishFeedingRenderer(env)

obs = env.reset()
if isinstance(obs, tuple):
    obs, _ = obs

frames = []
done = False

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    renderer.render(step=step, reward=reward)
    
    # Save frame
    frame_surface = pygame.display.get_surface()
    frame_array = pygame.surfarray.array3d(frame_surface)
    frame_array = frame_array.transpose([1, 0, 2])
    frames.append(frame_array)

    time.sleep(0.1)
    if done:
        break

renderer.close()

# Save as GIF
imageio.mimsave("fish_agent_random.gif", frames, duration=5)
