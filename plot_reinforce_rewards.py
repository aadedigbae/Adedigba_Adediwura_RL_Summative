import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# === CONFIGURE PATH ===
reinforce_log_dir = "logs/reinforce/A2C_1"  # Folder containing events.out.tfevents...
reinforce_tag = "rollout/ep_rew_mean"  # Name of scalar in TensorBoard

# === FUNCTION TO READ REINFORCE REWARDS FROM TENSORBOARD LOG ===
def load_tb_rewards(log_dir, tag):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    if tag not in event_acc.Tags()["scalars"]:
        raise ValueError(f"Tag '{tag}' not found in TensorBoard logs. Available: {event_acc.Tags()['scalars']}")
    
    scalar_events = event_acc.Scalars(tag)
    timesteps = [e.step for e in scalar_events]
    rewards = [e.value for e in scalar_events]
    return timesteps, rewards

# === LOAD REINFORCE DATA ===
timesteps, rewards = load_tb_rewards(reinforce_log_dir, reinforce_tag)

# === PLOT ===
plt.figure(figsize=(8, 5))
plt.plot(timesteps, rewards, color="purple", label="REINFORCE")
plt.xlabel("Timesteps")
plt.ylabel("Cumulative Reward")
plt.title("REINFORCE Cumulative Reward Over Time")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()

# Save & show
plt.savefig("reinforce_cumulative_rewards.png")
plt.show()

print(f"✅ Plot saved as reinforce_cumulative_rewards.png")
