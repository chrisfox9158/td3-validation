# Library imports
import numpy as np
import gymnasium as gym

# Local imports
from agent.td3 import TD3
from agent.noise_schedule import NoiseSchedule
from config import AGENT_SPECS, TRAINING_SPECS

env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_scale = float(env.action_space.high[0])  # Pendulum's range is [-2, 2]; actor outputs [-1, 1]

agent = TD3(obs_dim, action_dim, hardware_name="pendulum-v1")

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

env.close()