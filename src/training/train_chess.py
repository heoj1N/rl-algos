import matplotlib.pyplot as plt
from ..environments.chess_env import ChessEnv
from ..algorithms.chess_dqn import ChessDQN
from ..config.chess_dqn_config import CHESS_DQN_CONFIG

def main():
    # Training
    env = ChessEnv()
    agent = ChessDQN(env, CHESS_DQN_CONFIG)
    print("Starting training...")
    results = agent.train(num_episodes=1000)
    # Evaluation
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results['episode_rewards'])
    plt.title('Chess DQN Training Progress')
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
    plt.savefig('chess_training_results.png')
    print("\nEvaluating trained agent...")
    eval_results = agent.evaluate(num_episodes=10)
    print(f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    agent.save('data/checkpoints/chess_model.pth')
    env.close()

if __name__ == "__main__":
    main() 