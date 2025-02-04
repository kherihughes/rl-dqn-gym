import argparse
import gymnasium as gym
import os
import numpy as np

from src.dqn.agent import DQNAgent
from src.dqn.utils import plot_training_results, plot_evaluation_results

def parse_args():
    parser = argparse.ArgumentParser(description='Train DQN agent')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='environment name')
    parser.add_argument('--episodes', type=int, default=500,
                        help='number of episodes to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--buffer_size', type=int, default=10000,
                        help='replay buffer size')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--target_update_freq', type=int, default=100,
                        help='target network update frequency')
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='number of episodes for evaluation')
    parser.add_argument('--eval_freq', type=int, default=50,
                        help='frequency of evaluation during training')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='directory to save results')
    parser.add_argument('--load_path', type=str, default=None,
                        help='path to load a saved model')
    return parser.parse_args()

def evaluate_agent(agent, env, num_episodes):
    """Evaluate the agent for a number of episodes."""
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.q_network.get_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            
        rewards.append(episode_reward)
    
    return rewards

def main():
    # Parse arguments
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create environments
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq
    )
    
    # Load saved model if specified
    if args.load_path:
        print(f"Loading model from {args.load_path}")
        agent.load(args.load_path)
    
    # Train agent
    print("Starting training...")
    episode_rewards = []
    eval_results = []
    
    for episode in range(args.episodes):
        state = env.reset()[0]
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Store transition
            agent.replay_buffer.push(state, action, reward, float(done), next_state)
            
            # Train model
            loss = agent.train_step()
            
            # Update exploration rate
            agent.update_epsilon()
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        # Periodic evaluation
        if (episode + 1) % args.eval_freq == 0:
            eval_rewards = evaluate_agent(agent, eval_env, args.eval_episodes)
            avg_eval_reward = np.mean(eval_rewards)
            eval_results.append((episode + 1, avg_eval_reward))
            
            print(f"Episode {episode + 1}")
            print(f"  Training reward: {episode_reward:.2f}")
            print(f"  Average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
            print(f"  Evaluation reward: {avg_eval_reward:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            
            # Save if best
            agent.save_if_best(avg_eval_reward, args.save_dir)
    
    # Plot and save training results
    plot_training_results(episode_rewards, save_path=os.path.join(args.save_dir, 'training_results.png'))
    
    # Final evaluation
    print("\nFinal Evaluation...")
    final_eval_rewards = evaluate_agent(agent, eval_env, args.eval_episodes)
    avg_eval_reward = np.mean(final_eval_rewards)
    print(f"Final average evaluation reward: {avg_eval_reward:.2f}")
    
    # Plot and save evaluation results
    plot_evaluation_results(final_eval_rewards, save_path=os.path.join(args.save_dir, 'evaluation_results.png'))
    
    # Save final model
    agent.save(os.path.join(args.save_dir, 'final_model'))
    
    # Close environments
    env.close()
    eval_env.close()
    
if __name__ == "__main__":
    main() 