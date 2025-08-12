"""
plot_results.py
================
Plots cumulative rewards over episodes for all RL methods:
- A2C
- DQN
- PPO
- REINFORCE (read from TensorBoard log without retraining)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ======================
# CONFIGURATION
# ======================
LOG_PATHS = {
    "A2C": "logs/a2c/evaluations.npz",
    "DQN": "logs/dqn/evaluations.npz",
    "PPO": "logs/ppo/evaluations.npz",
}

REINFORCE_LOG_DIR = "reinforce/A2C_1"  # Folder containing REINFORCE events.out.tfevents...
REINFORCE_TAG = "rollout/ep_rew_mean"  # The scalar tag to read


# ======================
# HELPER FUNCTIONS
# ======================
def load_npz_evals(file_path):
    """
    Loads timesteps and mean rewards from evaluations.npz
    """
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return None, None
    data = np.load(file_path)
    timesteps = data["timesteps"]
    rewards = data["results"].mean(axis=1)  # mean over evaluation episodes
    return timesteps, rewards


def load_tb_scalar(log_dir, tag):
    """
    Loads scalar values from TensorBoard logs.
    """
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    if tag not in event_acc.Tags()["scalars"]:
        raise ValueError(f"Tag '{tag}' not found. Available: {event_acc.Tags()['scalars']}")

    events = event_acc.Scalars(tag)
    timesteps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return timesteps, values


# ======================
# MAIN PLOTTING
# ======================
def plot_cumulative_rewards():
    results = {}

    # Load standard agents (A2C, DQN, PPO)
    for name, path in LOG_PATHS.items():
        t, r = load_npz_evals(path)
        if t is not None:
            results[name] = (t, r)

    # Load REINFORCE from TensorBoard
    try:
        t_r, r_r = load_tb_scalar(REINFORCE_LOG_DIR, REINFORCE_TAG)
        results["REINFORCE"] = (t_r, r_r)
    except Exception as e:
        print(f"⚠️ Could not load REINFORCE data: {e}")

    # Plot all curves
    plt.figure(figsize=(10, 6))
    for name, (t, r) in results.items():
        plt.plot(t, r, label=name)

    plt.xlabel("Timesteps")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Rewards Over Time (All Methods)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cumulative_rewards_all_methods.png")
    plt.show()


if __name__ == "__main__":
    plot_cumulative_rewards()
