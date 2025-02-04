from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

def plot_training_results(
    episode_rewards: List[float],
    window: int = 100,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training rewards and moving average.
    
    Args:
        episode_rewards: List of episode rewards from training
        window: Window size for computing moving average. Will be adjusted if larger than data
        save_path: If provided, save the plot to this path instead of displaying
        
    The plot shows:
        - Raw episode rewards (blue line)
        - Moving average with specified window (red line)
        - Grid for better readability
        - Legend indicating raw rewards and moving average
    """
    plt.figure(figsize=(10, 6))
    
    # Plot episode rewards
    plt.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Adjust window size if needed
    window = min(window, len(episode_rewards))
    if window > 0:
        # Plot moving average
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, color='red', 
                label=f'Moving Average (window={window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_evaluation_results(
    eval_rewards: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    Plot evaluation rewards distribution.
    
    Args:
        eval_rewards: List of rewards from evaluation episodes
        save_path: If provided, save the plot to this path instead of displaying
        
    The plot shows:
        - Histogram of evaluation rewards (blue bars)
        - Mean reward as a vertical line (red dashed)
        - Grid for better readability
        - Legend showing the mean reward value
    """
    plt.figure(figsize=(10, 6))
    
    # Adjust number of bins based on data size
    num_bins = min(20, len(eval_rewards) // 5 + 1)
    plt.hist(eval_rewards, bins=num_bins, color='blue', alpha=0.7)
    mean_reward = np.mean(eval_rewards)
    plt.axvline(mean_reward, color='red', linestyle='dashed', 
                label=f'Mean: {mean_reward:.1f}')
    
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.title('Evaluation Rewards Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 