# Library imports
import numpy as np
import gymnasium as gym
import os
import json
import datetime
import matplotlib.pyplot as plt

# Local imports
from agent.td3 import TD3
from agent.noise_schedule import NoiseSchedule
from config import AGENT_SPECS, TRAINING_SPECS

# Run-logging
def export_run(episode_rewards, run_dir):
    """Save raw episode rewards as JSON, plus a convergence plot with rolling average."""
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({"episode_rewards": episode_rewards}, f, indent=2)

    window = 20
    rolling_avg = [
        np.mean(episode_rewards[max(0, i - window):i + 1])
        for i in range(len(episode_rewards))
    ]

    # Reward convergence plot
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.3, label="raw episode reward")
    plt.plot(rolling_avg, linewidth=2, label=f"{window}-episode rolling average")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("TD3 Convergence on Pendulum-v1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(run_dir, "convergence.png"), dpi=150)
    plt.close()

    # Distribution of the final 100 episodes
    plt.figure(figsize=(8, 5))
    final_stretch = episode_rewards[-100:]
    plt.hist(final_stretch, bins=30, edgecolor="black")
    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")
    plt.title(f"Reward Distribution — Final {len(final_stretch)} Episodes")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(run_dir, "final_distribution.png"), dpi=150)
    plt.close()

# Training loop
env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_scale = float(env.action_space.high[0])  # Pendulum's range is [-2, 2]; actor outputs [-1, 1]

agent = TD3(obs_dim, action_dim, hardware_name="pendulum-v1")

run_dir = os.path.join("runs", datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
episode_rewards = []

total_steps = 0
for episode in range(TRAINING_SPECS["num_episodes"]):
    obs, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    noise_scale = TRAINING_SPECS["noise_start"]

    while True:
        if total_steps < TRAINING_SPECS["warmup_steps"]:
            raw_action = np.random.uniform(-1, 1, size=action_dim)
        else:
            noise_scale = NoiseSchedule.exponential_noise_decay(episode, TRAINING_SPECS["noise_start"], TRAINING_SPECS["noise_end"], TRAINING_SPECS["noise_decay_rate"])
            raw_action = agent.select_action(obs, noise_scale=noise_scale)

        env_action = raw_action * action_scale  # scale [-1,1] actor output into Pendulum's [-2,2] range

        next_obs, reward, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated

        agent.replay_buffer.add(obs, raw_action, reward, next_obs, done)  # store the raw [-1,1] action only

        obs = next_obs
        episode_reward += reward
        total_steps += 1
        episode_steps += 1

        if agent.replay_buffer.size >= AGENT_SPECS["replay_buffer_batch_size"] and total_steps % TRAINING_SPECS["train_freq"] == 0:
            agent.update()

        if done:
            break

    print(f"Episode {episode}: reward= {episode_reward:.2f}, steps= {episode_steps}")
    episode_rewards.append(episode_reward)

export_run(episode_rewards, run_dir)
env.close()