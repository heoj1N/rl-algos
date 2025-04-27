"""Default configuration for Chess DQN algorithm."""

CHESS_DQN_CONFIG = {
    # Network parameters
    'learning_rate': 0.0001,
    
    # Replay buffer parameters
    'memory_size': 100000,  # Larger memory for chess
    'batch_size': 128,      # Larger batch size for stability
    
    # Training parameters
    'gamma': 0.99,          # Discount factor
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.999, # Slower decay for chess
    'target_update': 1000,  # Update target network less frequently
} 