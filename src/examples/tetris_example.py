import matplotlib.pyplot as plt
from ..environments.tetris_env import TetrisEnv
from ..algorithms.tetris_dqn import TetrisDQN
from ..config.tetris_dqn_config import TETRIS_DQN_CONFIG
import torch

def main():
    # Create Tetris environment
    env = TetrisEnv()
    
    # Initialize DQN agent
    agent = TetrisDQN(env, TETRIS_DQN_CONFIG)
    
    # Train the agent
    print("Starting training...")
    results = agent.train(num_episodes=1000)
    
    # Plot training results
    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(results['episode_rewards'])
    plt.title('Tetris DQN Training Progress')
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
    plt.savefig('tetris_training_results.png')
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    eval_results = agent.evaluate(num_episodes=10)
    print(f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    
    # Save the trained model
    agent.save('data/tetris_model.pth')
    
    # Demonstrate the trained agent
    print("\nDemonstrating trained agent...")
    state, _ = env.reset()
    total_reward = 0
    
    while True:
        env.render()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(agent.device)
            q_values = agent.policy_net(state_tensor)
            action = q_values.argmax().item()
        
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    print(f"Final score: {total_reward}")
    env.close()

if __name__ == "__main__":
    # This script should be run as a module:
    # python -m src.examples.tetris_example
    main() 