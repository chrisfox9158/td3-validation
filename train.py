# Library imports
import numpy as np
import os
import datetime

# Local imports
from env.gripper_env import GripperEnv
from env import observations, rewards
from agent.td3 import TD3
from agent.noise_schedule import NoiseSchedule
from training_logs.run_logger import RunLogger
from training_logs.run_export import export_run
from config import HARDWARE
from config import AGENT_SPECS
from config import TRAINING_SPECS

run_dir = os.path.join("runs", datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))

# Setup
env = GripperEnv(
    xml_path="three_finger_two_joint_gripper.xml",
    obs_extractors=[observations.obs_joint_angles, observations.obs_touch_sensors, observations.obs_joint_velocities],
    reward_terms=[rewards.reward_drop_penalty, rewards.reward_crush_penalty,
                  rewards.reward_grasp, rewards.reward_distance,
                  rewards.reward_lift, rewards.reward_success]
)

initial_obs = env.reset()
obs_dim = env.obs_dim
action_dim = env.action_dim
hardware_name = HARDWARE["hardware_name"]

agent = TD3(obs_dim, action_dim, hardware_name)
term_names = ["drop", "crush", "grasp", "distance", "lift", "success"]

# Episode loop
total_steps = 0
logger = RunLogger()
for episode in range(TRAINING_SPECS["num_episodes"]):
    obs = env.reset()
    episode_reward = 0
    episode_steps = 0
    term_totals = {name: 0.0 for name in term_names}
    noise_scale = TRAINING_SPECS["noise_start"]  # default during warmup, overwritten once policy-driven actions begin

    while True:
        if total_steps < TRAINING_SPECS["warmup_steps"]:
            action = np.random.uniform(-1, 1, size=action_dim)
        else:
            noise_scale = NoiseSchedule.exponential_noise_decay(episode, TRAINING_SPECS["noise_start"], TRAINING_SPECS["noise_end"], TRAINING_SPECS["noise_decay_rate"])
            action = agent.select_action(obs, noise_scale=noise_scale)

        next_obs, reward, done = env.step(action)
        for name, value in env.last_reward_breakdown.items():
            term_totals[name] = term_totals.get(name, 0.0) + value

        agent.replay_buffer.add(obs, action, reward, next_obs, done)

        obs = next_obs
        episode_reward += reward
        total_steps += 1
        episode_steps += 1

        if agent.replay_buffer.size >= AGENT_SPECS["replay_buffer_batch_size"] and total_steps % TRAINING_SPECS["train_freq"] == 0:
            agent.update()

        if done:
            break
    
    if episode % 100 == 0:
        export_run(logger, run_dir)
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        agent.save(os.path.join(checkpoint_dir, "checkpoint.pt"), os.path.join(checkpoint_dir, "metadata.json"), obs_dim, action_dim, hardware_name)

    logger.log_episode(episode, episode_reward, term_totals, episode_steps, env.info, noise_scale)

    term_str = "  ".join(f"{name}={term_totals[name]:.2f}" for name in term_names)
    print(f"Episode {episode}: reward= {episode_reward:.2f}, steps= {total_steps}")