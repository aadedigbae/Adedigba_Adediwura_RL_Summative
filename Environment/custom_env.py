# ✅ Precision Aquaculture Custom Environment
# This file defines a custom Gymnasium environment for a smart fish feeding agent.
# The agent interacts in a 5x5 grid fish tank where it decides when and where to feed fish.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class FishFeedingEnv(gym.Env):
    """
    A custom Gym environment simulating a smart fish tank where the agent:
    - Moves around the tank
    - Decides when to feed fish
    - Avoids overfeeding (which leads to penalties)
    """

    def __init__(self, render_mode=None):
        super(FishFeedingEnv, self).__init__()
        self.grid_size = 5  # Tank is 5x5
        self.max_steps = 50
        self.action_space = spaces.Discrete(6)  # 0=up, 1=down, 2=left, 3=right, 4=feed, 5=skip

        # Observation space: 5x5 grid of fish hunger levels (0 or 1), water quality, agent position
        # We'll use a flat array to represent the whole state
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.grid_size * self.grid_size + 3,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [0, 0]  # Start in top-left corner
        self.water_quality = 1.0  # Perfect water quality
        self.steps = 0

        # Fish hunger: 1 = hungry, 0 = not hungry
        self.fish_hunger = np.random.choice([0, 1], size=(self.grid_size, self.grid_size))

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        flat_hunger = self.fish_hunger.flatten()
        obs = np.concatenate((
            flat_hunger,
            [self.water_quality],
            [self.agent_pos[0] / (self.grid_size - 1)],
            [self.agent_pos[1] / (self.grid_size - 1)]
        ))
        return obs.astype(np.float32)

    def step(self, action):
        self.steps += 1
        reward = -1  # Default small penalty to discourage idle movement
        terminated = False

        # Move agent
        if action == 0 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1  # Up
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1  # Down
        elif action == 2 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1  # Left
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1  # Right
        elif action == 4:  # Feed
            x, y = self.agent_pos
            if self.fish_hunger[y][x] == 1:
                reward = 10  # Correct feeding
                self.fish_hunger[y][x] = 0
            else:
                reward = -5  # Overfeeding penalty
                self.water_quality -= 0.1
        elif action == 5:  # Skip feed
            x, y = self.agent_pos
            if self.fish_hunger[y][x] == 1:
                reward = -3  # Skipped feeding a hungry fish

        # Termination conditions
        if self.steps >= self.max_steps:
            terminated = True
        if self.water_quality <= 0.4:
            terminated = True
        if np.sum(self.fish_hunger) == 0:
            terminated = True  # All fish are fed

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.fish_hunger[y][x] == 1:
                    grid[y][x] = 'F'  # Hungry fish
        ax, ay = self.agent_pos
        grid[ay][ax] = 'A'  # Agent

        print("Tank State (F = Hungry Fish, A = Agent)")
        for row in grid:
            print(' '.join(row))
        print(f"Water Quality: {self.water_quality:.2f}\n")
