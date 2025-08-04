from stable_baselines3.common.callbacks import BaseCallback

class CurriculumCallback(BaseCallback):
    def __init__(self, update_every=50000, max_fish_count=10, verbose=1):
        super().__init__(verbose)
        self.update_every = update_every
        self.max_fish_count = max_fish_count

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_every == 0:
            env = self.training_env.envs[0]
            if hasattr(env, "increase_difficulty"):
                new_count = env.current_fish_count + 1
                env.increase_difficulty(new_count)
        return True
