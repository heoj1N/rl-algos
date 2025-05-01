import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.environments.tetris_env import TetrisEnv
from src.algorithms.tetris_dqn import TetrisDQN
from src.config.tetris_dqn_config import TETRIS_DQN_CONFIG
import torch

def main():
    # Training
    env = TetrisEnv()
    agent = TetrisDQN(env, TETRIS_DQN_CONFIG)
    print("Starting training...")
    results = agent.train(num_episodes=1000)
    # Evaluation
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results['episode_rewards'])
    plt.title('Tetris DQN Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(results['losses'])
    plt.title('Training Loss')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('tetris_training_results.png')
    print("\nEvaluating trained agent...")
    eval_results = agent.evaluate(num_episodes=10)
    print(f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    agent.save('data/checkpoints/tetris_model.pth')
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
    main() 