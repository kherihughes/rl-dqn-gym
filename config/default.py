"""Default configuration for DQN training."""

# Environment settings
ENV_NAME = "CartPole-v1"
MAX_STEPS = 1000

# Training settings
NUM_EPISODES = 300
EVAL_EPISODES = 100
EVAL_FREQ = 50

# Model architecture
HIDDEN_DIMS = [128, 128]

# DQN hyperparameters
LEARNING_RATE = 5e-4
GAMMA = 0.99
BUFFER_SIZE = 20000
BATCH_SIZE = 128
TARGET_UPDATE_FREQ = 100

# Exploration settings
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.997

# Device settings
DEVICE = "cuda"  # or "cpu"

# Paths
SAVE_DIR = "results"
MODEL_NAME = "dqn_cartpole" 