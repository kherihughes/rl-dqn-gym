# Deep Q-Network (DQN) Implementation

This repository contains a PyTorch implementation of the Deep Q-Network (DQN) algorithm for reinforcement learning. The implementation includes experience replay and target networks as described in the original DQN paper.

## Features

- Deep Q-Network implementation using PyTorch
- Experience Replay Buffer
- Target Network for stable training
- Configurable hyperparameters
- Support for OpenAI Gymnasium environments
- Training progress visualization
- Model saving and loading
- Best model checkpointing

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

To train a DQN agent on the CartPole environment:

```bash
python scripts/train.py
```

You can modify the hyperparameters using command-line arguments:

```bash
# Example with custom parameters
python scripts/train.py --env CartPole-v1 --episodes 1000 --eval_freq 100

# Continue training from a saved model
python scripts/train.py --load_path results/best_model/agent_state.pt
```

Available arguments:
- `--env`: Environment name (default: CartPole-v1)
- `--episodes`: Number of training episodes (default: 300)
- `--lr`: Learning rate (default: 5e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--buffer_size`: Replay buffer size (default: 20000)
- `--batch_size`: Training batch size (default: 128)
- `--target_update_freq`: Target network update frequency (default: 100)
- `--eval_episodes`: Number of evaluation episodes (default: 100)
- `--eval_freq`: Frequency of evaluation during training (default: 50)
- `--save_dir`: Directory to save results (default: 'results')
- `--load_path`: Path to load a saved model (optional)

## Project Structure

```
.
├── src/
│   └── dqn/
│       ├── __init__.py
│       ├── agent.py      # DQN agent implementation
│       ├── model.py      # Neural network architecture
│       ├── replay_buffer.py  # Experience replay implementation
│       └── utils.py      # Utility functions and plotting
├── scripts/
│   └── train.py         # Training script
├── tests/
│   └── test_dqn.py      # Unit tests
├── results/             # Training results and plots
├── requirements.txt     # Project dependencies
├── LICENSE             # MIT License
└── README.md
```

## Results

The DQN agent achieves optimal performance on the CartPole-v1 environment, consistently reaching the maximum score of 500 during evaluation. Here are the key results from our latest training run:

- Training Progress:
  - Reaches optimal performance (500 reward) by episode 200
  - Maintains stable performance throughout training
  - Final evaluation reward: 500.0 (perfect score)

Training Characteristics:
- Fast initial learning (first 200 episodes)
- Stable performance after convergence
- Consistent evaluation scores of 500.0
- Efficient exploration with epsilon decay

Default Hyperparameters:
```python
learning_rate = 5e-4
gamma = 0.99
buffer_size = 20000
batch_size = 128
target_update_freq = 100
epsilon_decay = 0.997
hidden_dims = [128, 128]
```

## Implementation Details

The implementation includes several key DQN features:
- Double Q-learning with target network
- Experience replay for sample efficiency
- Epsilon-greedy exploration with decay
- Efficient batched training on GPU when available
- Best model checkpointing
- Periodic evaluation

Network Architecture:
- Input: State dimension (4 for CartPole)
- Hidden layers: [128, 128] with ReLU activation
- Output: Action values (2 for CartPole)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) 