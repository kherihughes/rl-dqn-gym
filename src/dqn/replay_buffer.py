from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, done, next_state):
        """
        Store transition in the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            done: Whether episode ended
            next_state: Next state
        """
        self.buffer.append((state, action, reward, done, next_state))
        
    def sample(self, batch_size: int):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, dones, next_states) as tensors
        """
        transitions = random.sample(self.buffer, batch_size)
        
        # Transpose the batch of transitions
        batch = list(zip(*transitions))
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.LongTensor(batch[1])
        rewards = torch.FloatTensor(batch[2])
        dones = torch.FloatTensor(batch[3])
        next_states = torch.FloatTensor(np.array(batch[4]))
        
        return states, actions, rewards, dones, next_states
    
    def __len__(self):
        """Return current size of the buffer."""
        return len(self.buffer) 