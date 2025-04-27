import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any

# Tetris piece shapes
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 1], [1, 0, 0]],  # L
    [[1, 1, 1], [0, 0, 1]],  # J
    [[1, 1, 0], [0, 1, 1]],  # S
    [[0, 1, 1], [1, 1, 0]]   # Z
]

class TetrisEnv(gym.Env):
    """Tetris environment implementing the gymnasium interface."""
    
    def __init__(self, width=10, height=20):
        super().__init__()
        
        self.width = width
        self.height = height
        
        # Define action space (0: left, 1: right, 2: rotate, 3: drop)
        self.action_space = spaces.Discrete(4)
        
        # Define observation space (board state + current piece)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(height, width), dtype=np.int8
        )
        
        # Initialize game state
        self.board = np.zeros((height, width), dtype=np.int8)
        self.current_piece = None
        self.current_pos = None
        self.current_rotation = 0
        
        # Game state
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
    
    def _get_new_piece(self):
        """Generate a new piece at the top of the board."""
        piece_idx = np.random.randint(len(SHAPES))
        self.current_piece = np.array(SHAPES[piece_idx])
        self.current_rotation = 0
        self.current_pos = [0, self.width // 2 - len(self.current_piece[0]) // 2]
    
    def _rotate_piece(self):
        """Rotate the current piece 90 degrees clockwise."""
        self.current_piece = np.rot90(self.current_piece, -1)
        self.current_rotation = (self.current_rotation + 1) % 4
    
    def _is_valid_position(self, piece, pos):
        """Check if the piece is in a valid position."""
        piece_height, piece_width = piece.shape
        for y in range(piece_height):
            for x in range(piece_width):
                if piece[y, x]:
                    board_y = pos[0] + y
                    board_x = pos[1] + x
                    if (board_y >= self.height or 
                        board_x < 0 or 
                        board_x >= self.width or 
                        (board_y >= 0 and self.board[board_y, board_x])):
                        return False
        return True
    
    def _place_piece(self):
        """Place the current piece on the board."""
        piece_height, piece_width = self.current_piece.shape
        for y in range(piece_height):
            for x in range(piece_width):
                if self.current_piece[y, x]:
                    board_y = self.current_pos[0] + y
                    board_x = self.current_pos[1] + x
                    if board_y >= 0:
                        self.board[board_y, board_x] = 1
    
    def _clear_lines(self):
        """Clear completed lines and update score."""
        lines_to_clear = []
        for y in range(self.height):
            if np.all(self.board[y]):
                lines_to_clear.append(y)
        
        if lines_to_clear:
            self.lines_cleared += len(lines_to_clear)
            self.score += len(lines_to_clear) * 100
            
            # Remove cleared lines
            for line in sorted(lines_to_clear, reverse=True):
                self.board = np.vstack((np.zeros((1, self.width)), self.board[:line], self.board[line+1:]))
    
    def reset(self, *, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        self._get_new_piece()
        
        return self.board.copy(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        if self.game_over:
            return self.board.copy(), 0, True, False, {}
        
        # Apply action
        if action == 0:  # Move left
            new_pos = [self.current_pos[0], self.current_pos[1] - 1]
            if self._is_valid_position(self.current_piece, new_pos):
                self.current_pos = new_pos
        elif action == 1:  # Move right
            new_pos = [self.current_pos[0], self.current_pos[1] + 1]
            if self._is_valid_position(self.current_piece, new_pos):
                self.current_pos = new_pos
        elif action == 2:  # Rotate
            original_piece = self.current_piece.copy()
            self._rotate_piece()
            if not self._is_valid_position(self.current_piece, self.current_pos):
                self.current_piece = original_piece
                self.current_rotation = (self.current_rotation - 1) % 4
        
        # Move piece down
        new_pos = [self.current_pos[0] + 1, self.current_pos[1]]
        if self._is_valid_position(self.current_piece, new_pos):
            self.current_pos = new_pos
        else:
            # Place piece and check for game over
            self._place_piece()
            self._clear_lines()
            
            # Check if game is over
            if self.current_pos[0] <= 0:
                self.game_over = True
                return self.board.copy(), -100, True, False, {}
            
            # Get new piece
            self._get_new_piece()
            if not self._is_valid_position(self.current_piece, self.current_pos):
                self.game_over = True
                return self.board.copy(), -100, True, False, {}
        
        # Calculate reward
        reward = 0
        if action == 3:  # Hard drop
            while self._is_valid_position(self.current_piece, [self.current_pos[0] + 1, self.current_pos[1]]):
                self.current_pos[0] += 1
            reward = 2  # Bonus for hard drop
        
        return self.board.copy(), reward, self.game_over, False, {}
    
    def render(self) -> None:
        """Render the current board state."""
        display_board = self.board.copy()
        if not self.game_over:
            piece_height, piece_width = self.current_piece.shape
            for y in range(piece_height):
                for x in range(piece_width):
                    if self.current_piece[y, x]:
                        board_y = self.current_pos[0] + y
                        board_x = self.current_pos[1] + x
                        if 0 <= board_y < self.height and 0 <= board_x < self.width:
                            display_board[board_y, board_x] = 2
        
        for row in display_board:
            print(''.join(['□' if cell == 0 else '■' if cell == 1 else '▣' for cell in row]))
        print(f"Score: {self.score}, Lines: {self.lines_cleared}") 