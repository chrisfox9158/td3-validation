# Library imports
import numpy as np

# Main class
class NoiseSchedule():
    """Exploration noise decay schedules for training."""

    @staticmethod
    def exponential_noise_decay(episode, start, end, decay_rate):
        """Exponentially decay noise from `start` toward `end`, controlled by `decay_rate`."""
        return end + (start - end) * np.exp(-decay_rate * episode)
    
    @staticmethod
    def linear_noise_decay(episode, start, end, decay_episodes):
        """Linearly decay noise from `start` to `end` over `decay_episodes`, then hold at `end`."""
        fraction = min(episode / decay_episodes, 1.0)
        return start - fraction * (start - end)