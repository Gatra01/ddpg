from env import PowerAllocationEnv
from ddpg import DDPGAgent
import numpy as np

def train_ddpg():
    num_episodes = 500
    max_steps = 100
    batch_size = 64
    state_dim = 10  
    action_dim = 5  # Power allocation for 5 nodes
    max_action = 1  # Power max allocation

    env = PowerAllocationEnv(num_nodes=5, p_max=1, noise_power=0.01)
    agent = DDPGAgent(state_dim, action_dim, max_action)

    for episode in range(num_episodes):
        state = np.array(env.reset()).flatten()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            env.power = action  # Set new power allocation
            env.hitung_data_rate()  # Update SINR & data rate
            
            next_state = np.array(env.observation()).flatten()
            reward = env.hitung_efisiensi_energi()
            done = False  # No termination in this environment
            
            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.train(batch_size)

            state = next_state
            episode_reward += reward
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.4f}")

    print("Training selesai!")

if __name__ == "__main__":
    train_ddpg()
