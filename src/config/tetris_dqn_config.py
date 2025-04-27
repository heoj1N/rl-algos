TETRIS_DQN_CONFIG = {
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'target_update': 10,
    'memory_size': 10000,
    'batch_size': 64,
    'hidden_size': 128,
    'device': 'cuda'  # Change to 'cpu' if no GPU is available
} 