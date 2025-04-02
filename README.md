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

Available arguments (defaults shown):
- `--env`: Environment name (default: `CartPole-v1`)
- `--episodes`: Number of training episodes (default: 500)
- `--lr`: Learning rate (default: 1e-3)
- `--gamma`: Discount factor (default: 0.99)
- `--buffer_size`: Replay buffer size (default: 10000)
- `--batch_size`: Training batch size (default: 64)
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

The DQN agent is capable of achieving optimal performance on the CartPole-v1 environment, reaching the maximum score of 500 during evaluation in typical runs.

- Training Progress Example:
  - Often reaches near-optimal performance (average reward > 475) within the default 500 episodes.
  - Maintains stable performance after convergence.
  - Example final evaluation reward (average over 100 episodes): ~490-500

Training Characteristics:
- Relatively fast initial learning.
- Stable performance after convergence.

Effective Default Hyperparameters (from `scripts/train.py`):
```python
environment = 'CartPole-v1'
episodes = 500
learning_rate = 1e-3
gamma = 0.99
buffer_size = 10000
batch_size = 64
target_update_freq = 100
eval_episodes = 100
eval_freq = 50
# Note: Epsilon decay parameters and hidden_dims are defined within DQNAgent
# epsilon_start = 1.0, epsilon_end = 0.01, epsilon_decay = 0.997
# hidden_dims = [128, 128]
```

## Implementation Details

The implementation includes several key DQN features:
- Standard DQN loss calculation with a target network for stability.
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