import unittest
import numpy as np
import torch
import gym

from src.dqn import DQNAgent, QNetwork, ReplayBuffer

class TestDQN(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('CartPole-v0')
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
    def test_qnetwork(self):
        network = QNetwork(self.state_dim, self.action_dim)
        state = torch.randn(1, self.state_dim)
        output = network(state)
        
        self.assertEqual(output.shape, (1, self.action_dim))
        
    def test_replay_buffer(self):
        buffer = ReplayBuffer(100)
        state = np.random.randn(self.state_dim)
        next_state = np.random.randn(self.state_dim)
        
        buffer.push(state, 0, 1.0, False, next_state)
        self.assertEqual(len(buffer), 1)
        
    def test_agent_initialization(self):
        agent = DQNAgent(self.state_dim, self.action_dim)
        self.assertEqual(agent.state_dim, self.state_dim)
        self.assertEqual(agent.action_dim, self.action_dim)
        
    def tearDown(self):
        self.env.close()
        
if __name__ == '__main__':
    unittest.main() 