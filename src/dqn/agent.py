import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from .model import QNetwork
from .replay_buffer import ReplayBuffer

class DQNAgent:
    """DQN Agent implementation."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        buffer_size: int = 20000,
        batch_size: int = 128,
        target_update_freq: int = 100,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.997,
        hidden_dims: list = [128, 128],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor
            buffer_size (int): Size of replay buffer
            batch_size (int): Size of training batch
            target_update_freq (int): Frequency of target network updates
            epsilon_start (float): Initial exploration rate
            epsilon_end (float): Final exploration rate
            epsilon_decay (float): Rate of exploration decay
            hidden_dims (list): Hidden layer dimensions
            device (str): Device to run the model on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.hidden_dims = hidden_dims
        self.lr = learning_rate
        
        print(f"Using device: {device}")
        
        # Initialize networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims, device=device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims, device=device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.total_steps = 0
        self.best_eval_reward = float('-inf')
        
    def save(self, save_dir: str):
        """
        Save the agent's state.
        
        Args:
            save_dir (str): Directory to save the agent state
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model state
        state = {
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': float(self.epsilon),  # Convert to native Python float
            'total_steps': int(self.total_steps),  # Convert to native Python int
            'best_eval_reward': float(self.best_eval_reward),  # Convert to native Python float
            # Save hyperparameters
            'state_dim': int(self.state_dim),
            'action_dim': int(self.action_dim),
            'hidden_dims': list(map(int, self.hidden_dims)),  # Convert to native Python ints
            'lr': float(self.lr),
            'gamma': float(self.gamma),
        }
        torch.save(state, os.path.join(save_dir, 'agent_state.pt'))
        
    def load(self, load_path: str):
        """
        Load the agent's state.
        
        Args:
            load_path (str): Path to the saved agent state
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Verify model architecture matches
        assert self.state_dim == checkpoint['state_dim'], "State dimension mismatch"
        assert self.action_dim == checkpoint['action_dim'], "Action dimension mismatch"
        assert self.hidden_dims == checkpoint['hidden_dims'], "Network architecture mismatch"
        
        # Load model state
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Load other parameters
        self.epsilon = float(checkpoint['epsilon'])
        self.total_steps = int(checkpoint['total_steps'])
        self.best_eval_reward = float(checkpoint['best_eval_reward'])
        
    def save_if_best(self, eval_reward: float, save_dir: str):
        """
        Save the model if it achieves a new best evaluation reward.
        
        Args:
            eval_reward (float): Current evaluation reward
            save_dir (str): Directory to save the model
        """
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            self.save(os.path.join(save_dir, 'best_model'))
            print(f"New best model saved with reward: {eval_reward:.2f}")
        
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        return self.q_network.get_action(state, self.epsilon)
    
    def update_epsilon(self):
        """Update exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def train_step(self):
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
            
        # Sample batch from replay buffer
        states, actions, rewards, dones, next_states = self.replay_buffer.sample(self.batch_size)
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        next_states = next_states.to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network if needed
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return loss.item()
    
    def train(self, env, num_episodes: int, max_steps: int = 1000):
        """
        Train the agent.
        
        Args:
            env: Training environment
            num_episodes (int): Number of episodes to train for
            max_steps (int): Maximum steps per episode
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()[0]  # Updated to handle new gym API
            episode_reward = 0
            episode_loss = 0
            num_updates = 0
            
            for step in range(max_steps):
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)  # Updated to handle new gym API
                done = terminated or truncated  # Combine both signals
                episode_reward += reward
                
                # Store transition
                self.replay_buffer.push(state, action, reward, float(done), next_state)
                
                # Train model
                loss = self.train_step()
                if loss is not None:
                    episode_loss += loss
                    num_updates += 1
                
                # Update exploration rate
                self.update_epsilon()
                
                state = next_state
                if done:
                    break
            
            # Log metrics
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else episode_reward
            avg_loss = episode_loss / num_updates if num_updates > 0 else 0
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}, Average Reward (last 100): {avg_reward:.2f}")
                
        return episode_rewards
    
    def evaluate(self, env, num_episodes: int):
        """
        Evaluate the agent.
        
        Args:
            env: Evaluation environment
            num_episodes (int): Number of episodes to evaluate for
            
        Returns:
            Average reward over episodes
        """
        rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()[0]  # Updated to handle new gym API
            episode_reward = 0
            done = False
            
            while not done:
                action = self.q_network.get_action(state, epsilon=0.0)  # No exploration during evaluation
                next_state, reward, terminated, truncated, _ = env.step(action)  # Updated to handle new gym API
                done = terminated or truncated  # Combine both signals
                episode_reward += reward
                state = next_state
                
            rewards.append(episode_reward)
            
        return np.mean(rewards) 