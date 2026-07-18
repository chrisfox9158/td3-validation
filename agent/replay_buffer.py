# Library imports
import numpy as np
import torch

# Local imports
from config import AGENT_SPECS

class ReplayBuffer():
    """Replay buffer system allowing random sampling of a large training-data pool."""
    def __init__(self, obs_dim, action_dim):
        self.capacity = AGENT_SPECS["replay_buffer_capacity"]
        self.batch_size = AGENT_SPECS["replay_buffer_batch_size"]
        self.size = 0
        self.ptr = 0
        
        self.states = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        i = np.random.choice(self.size, self.batch_size, replace=False)

        tensor_states = torch.from_numpy(self.states[i])
        tensor_actions = torch.from_numpy(self.actions[i])
        tensor_rewards = torch.from_numpy(self.rewards[i])
        tensor_next_states = torch.from_numpy(self.next_states[i])
        tensor_dones = torch.from_numpy(self.dones[i])

        return tensor_states, tensor_actions, tensor_rewards, tensor_next_states, tensor_dones