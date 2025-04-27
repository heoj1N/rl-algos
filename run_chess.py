"""Simple script to run the chess DQN example."""

import matplotlib.pyplot as plt
from src.environments.chess_env import ChessEnv
from src.algorithms.chess_dqn import ChessDQN
from src.config.chess_dqn_config import CHESS_DQN_CONFIG

def main():
    # Create chess environment
    env = ChessEnv()
    
    # Initialize DQN agent
    agent = ChessDQN(env, CHESS_DQN_CONFIG)
    
    # Train the agent
    print("Starting training...")
    results = agent.train(num_episodes=1000)
    
    # Plot training results
    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(results['episode_rewards'])
    plt.title('Chess DQN Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Plot losses
    plt.subplot(1, 2, 2)
    plt.plot(results['losses'])
    plt.title('Training Loss')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('chess_training_results.png')
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    eval_results = agent.evaluate(num_episodes=10)
    print(f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    
    # Save the trained model
    agent.save('chess_dqn_model.pth')
    
    env.close()

if __name__ == "__main__":
    main() 