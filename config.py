AGENT_SPECS = {
    "replay_buffer_capacity": int(1e6), # requires integer
    "replay_buffer_batch_size": int(256), # requires integer
    "actor_learning_rate": 3e-4, # known default: 3e-4
    "critic_learning_rate": 3e-4, # known default: 3e-4
    "default_noise_scale": 0.125,
    "gamma": 0.99, # known default: 0.99
    "policy_noise": 0.2, # known default: 0.2
    "noise_clip": 0.5, # known default: 0.5
    "policy_delay": 2, # known default: 2
    "tau": 0.005 # known default: 0.005
}

TRAINING_SPECS = {
    "num_episodes": 2000,
    "warmup_steps": 5000,
    "noise_start": 0.3,
    "noise_end": 0.05,
    "noise_decay_rate": 0.002,
    "train_freq": 4
}