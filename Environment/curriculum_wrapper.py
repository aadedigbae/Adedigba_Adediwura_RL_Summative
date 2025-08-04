import gymnasium as gym
import numpy as np

class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, initial_fish_count=2, max_fish_count=10):
        super().__init__(env)
        self.initial_fish_count = initial_fish_count
        self.max_fish_count = max_fish_count
        self.current_fish_count = initial_fish_count
        self.update_fish_count(self.current_fish_count)

    def update_fish_count(self, count):
        if hasattr(self.env, 'set_fish_count'):
            self.env.set_fish_count(count)
        else:
            print("[WARNING] Your base environment does not support setting fish count!")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.update_fish_count(self.current_fish_count)
        return obs, info

    def increase_difficulty(self, new_count):
        new_count = min(self.max_fish_count, new_count)
        if new_count != self.current_fish_count:
            print(f"🧪 Curriculum: Changing fish count to {new_count}")
            self.current_fish_count = new_count
            self.update_fish_count(new_count)
