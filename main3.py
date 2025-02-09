import numpy as np
import torch
import gym
from ddpg_pytorch import DDPG  # Pastikan file DDPG sudah dibuat dan bisa di-import
from power_allocation_env import PowerAllocationEnv  # Pastikan environment sudah dibuat
import matplotlib.pyplot as plt

# Hyperparameter Training
num_episodes = 500
timesteps = 200
batch_size = 64

# Logging Metrics
episode_rewards = []
energy_efficiency_log = []
data_rate_log = []
power_used_log = []

env = PowerAllocationEnv(num_devices=3, max_power=1.0, noise_power=0.1)
state_dim = env.state_dim
action_dim = env.action_dim
max_action = env.max_power  
ddpg_agent = DDPG(state_dim, action_dim, max_action)

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    total_data_rate = 0
    total_power_used = 0

    for t in range(timesteps):
        action = ddpg_agent.select_action(state)  # Pilih aksi dari actor
        next_state, reward, done, _ = env.step(action)
        
        # Simpan ke buffer
        ddpg_agent.replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        # Logging
        total_data_rate += np.sum(env.data_rate)
        total_power_used += np.sum(action)

        ddpg_agent.train(batch_size)  # Training

        if done:
            break

    # Simpan hasil per episode
    episode_rewards.append(episode_reward)
    energy_efficiency_log.append(episode_reward)  # Reward = Energy Efficiency
    data_rate_log.append(total_data_rate / timesteps)  # Rata-rata data rate
    power_used_log.append(total_power_used / timesteps)  # Rata-rata power used

    print(f"Episode {episode + 1} | Reward: {episode_reward:.3f} | Avg Data Rate: {data_rate_log[-1]:.3f} | Avg Power Used: {power_used_log[-1]:.3f}")

plt.figure(figsize=(10, 5))
plt.plot(energy_efficiency_log, label="Energy Efficiency", color='blue')
plt.xlabel("Episode")
plt.ylabel("Energy Efficiency")
plt.title("Energy Efficiency vs. Episode")
plt.legend()
plt.grid()
plt.show()
