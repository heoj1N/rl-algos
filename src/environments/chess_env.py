import chess
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any

class ChessEnv(gym.Env):
    def __init__(self):
        """Chess environment implementing the gymnasium interface."""
        super().__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(4672) # max legal moves
        self.observation_space = spaces.Box(
            low=-6, high=6, shape=(8, 8), dtype=np.int8
        )
        self.piece_to_int = {
            chess.PAWN: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.ROOK: 4,
            chess.QUEEN: 5,
            chess.KING: 6
        }
        self._create_move_mapping()
    
    def _create_move_mapping(self) -> None:
        """Create mapping between move indices and chess moves."""
        self.move_to_idx = {}
        self.idx_to_move = {}
        
        idx = 0
        for move in self.board.legal_moves:
            self.move_to_idx[move] = idx
            self.idx_to_move[idx] = move
            idx += 1
    
    def _board_to_observation(self) -> np.ndarray:
        """Convert chess board to observation array."""
        observation = np.zeros((8, 8), dtype=np.int8)
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = self.piece_to_int[piece.piece_type]
                if piece.color == chess.BLACK:
                    value = -value
                row, col = divmod(square, 8)
                observation[row, col] = value
        
        return observation
    
    def reset(self, *, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.board.reset()
        self._create_move_mapping()
        return self._board_to_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        if action not in self.idx_to_move:
            return self._board_to_observation(), -10, True, False, {}
        
        move = self.idx_to_move[action]
        self.board.push(move)
        
        # Check if game is over
        is_terminated = self.board.is_game_over()
        is_truncated = False
        
        # Calculate reward
        if is_terminated:
            if self.board.is_checkmate():
                reward = 1.0  # Win
            elif self.board.is_stalemate():
                reward = 0.0  # Draw
            else:
                reward = -1.0  # Loss
        else:
            # Simple material-based reward
            reward = self._calculate_material_reward()
        
        # Update move mapping
        self._create_move_mapping()
        
        return self._board_to_observation(), reward, is_terminated, is_truncated, {}
    
    def _calculate_material_reward(self) -> float:
        """Calculate reward based on material difference."""
        material_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = material_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        return (white_material - black_material) / 100.0  # Normalize reward
    
    def render(self) -> None:
        """Render the current board state."""
        print(self.board) 