# Library imports
import torch
import numpy as np

# Local imports
from agent.networks import Actor
from agent.networks import Critic
from agent.replay_buffer import ReplayBuffer
from config import AGENT_SPECS

class TD3():
    """The primary agent system: active agent instances, random noise injection, TD3-specific architecture."""
    def __init__(self, obs_dim, action_dim, hardware_name):
        """Set up TD3's six agents, two optimizers, and sync systems."""
        self.actor = Actor(obs_dim, action_dim)
        self.actor_target = Actor(obs_dim, action_dim)

        self.critic1 = Critic(obs_dim, action_dim)
        self.critic1_target = Critic(obs_dim, action_dim)
        self.critic2 = Critic(obs_dim, action_dim)
        self.critic2_target = Critic(obs_dim, action_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=AGENT_SPECS["actor_learning_rate"])
        self.critic_optimizer = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=AGENT_SPECS["critic_learning_rate"])

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.replay_buffer = ReplayBuffer(obs_dim, action_dim)
        self.step_counter = 0

    def select_action(self, state, noise_scale=AGENT_SPECS["default_noise_scale"]):
        """Select action and add random noise."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]

        noise = np.random.normal(0, noise_scale, size=action.shape)
        action = action + noise

        return np.clip(action, -1, 1)
    
    def update(self):
        """Sample batch, compute target Q-values; critic examinations; apply delayed actor update."""
        # Sample batch and compute target Q-values
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        with torch.no_grad():
            next_action = self.actor_target(next_states)
            noise = torch.randn_like(next_action) * AGENT_SPECS["policy_noise"]
            noise = torch.clamp(noise, -AGENT_SPECS["noise_clip"], AGENT_SPECS["noise_clip"])

            next_action = torch.clamp(next_action + noise, -1, 1)
        
            pessimistic_target_q = torch.min(self.critic1_target(next_states, next_action), self.critic2_target(next_states, next_action))
            target_Q = rewards + AGENT_SPECS["gamma"] * (1 - dones) * pessimistic_target_q

        # Critic examinations and the learning moment
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss = torch.nn.functional.mse_loss(current_q1, target_Q) + torch.nn.functional.mse_loss(current_q2, target_Q)

        self.critic_optimizer.zero_grad() # Clear old gradients for every parameter
        critic_loss.backward() # Calculate new gradients by chain rule for every parameter
        self.critic_optimizer.step() # Apply the optimizer update rule

        # Delayed actor update, after step counter increment
        self.step_counter += 1
        if self.step_counter % (AGENT_SPECS["policy_delay"]) == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean() # Average critic1's current states with actor's current states, as a minimized loss

            self.actor_optimizer.zero_grad() # Clear old gradients for every parameter
            actor_loss.backward() # Calculate new gradients by chain rule for every parameter
            self.actor_optimizer.step() # Apply the optimizer update rule

            # Soft target updates
            for target_param, live_param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(AGENT_SPECS["tau"] * live_param.data + (1 - AGENT_SPECS["tau"]) * target_param.data)
            
            for target_param, live_param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(AGENT_SPECS["tau"] * live_param.data + (1 - AGENT_SPECS["tau"]) * target_param.data)
            
            for target_param, live_param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(AGENT_SPECS["tau"] * live_param.data + (1 - AGENT_SPECS["tau"]) * target_param.data)