import gymnasium as gym
import matplotlib.pyplot as plt
from src.algorithms.dqn import DQN
from src.config.dqn_config import DQN_CONFIG

def main():
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Initialize DQN agent
    agent = DQN(env, DQN_CONFIG)
    
    # Train the agent
    print("Starting training...")
    results = agent.train(num_episodes=200)
    
    # Plot training results
    plt.figure(figsize=(10, 5))
    plt.plot(results['episode_rewards'])
    plt.title('DQN Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('training_results.png')
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    eval_results = agent.evaluate(num_episodes=10)
    print(f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    
    # Save the trained model
    agent.save('dqn_model.pth')
    
    env.close()

if __name__ == "__main__":
    main() 