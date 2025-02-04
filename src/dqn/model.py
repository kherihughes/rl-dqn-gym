import torch
import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    """Deep Q-Network for approximating Q-values."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 64], device: str = "cpu"):
        """
        Initialize the Q-Network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dims (list): List of hidden layer dimensions
            device (str): Device to run the model on
        """
        super(QNetwork, self).__init__()
        
        # Store device
        self.device = device
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Move model to device
        self.to(device)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for each action
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        return self.model(state)
    
    def get_action(self, state, epsilon=0.0):
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon (float): Exploration rate
            
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.model[-1].out_features)
        
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            state = state.unsqueeze(0) if len(state.shape) == 1 else state
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()
            
    def copy_weights_from(self, other):
        """
        Copy weights from another network.
        
        Args:
            other: Source network to copy weights from
        """
        for param, target_param in zip(self.parameters(), other.parameters()):
            param.data.copy_(target_param.data) 