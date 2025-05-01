import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QListWidget, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QFont
import torch
import numpy as np
from typing import Optional
from environments.tetris_env import TetrisEnv, SHAPES
from algorithms.tetris_dqn import TetrisDQN
from config.tetris_dqn_config import TETRIS_DQN_CONFIG

class TetrisBoardWidget(QWidget):
    """Custom widget for displaying the Tetris board."""
    
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setMinimumSize(300, 600)  # Standard Tetris board size
        self.block_size = 30  # Size of each block in pixels
        self.colors = {
            0: QColor(0, 0, 0),        # Empty
            1: QColor(0, 255, 255),    # Placed pieces
            2: QColor(255, 165, 0)     # Active piece
        }
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def keyPressEvent(self, event):
        """Forward key events to main window."""
        self.main_window.keyPressEvent(event)
    
    def focusInEvent(self, event):
        """Handle focus events."""
        super().focusInEvent(event)
        self.update()
    
    def mousePressEvent(self, event):
        """Set focus on click."""
        self.setFocus()
        super().mousePressEvent(event)
    
    def paintEvent(self, event):
        """Paint the Tetris board."""
        painter = QPainter(self)
        
        for y in range(20):
            for x in range(10):
                color = self.colors[self.main_window.env.board[y][x]]
                painter.fillRect(x * self.block_size, y * self.block_size,
                               self.block_size, self.block_size, color)
                painter.setPen(QColor(50, 50, 50))
                painter.drawRect(x * self.block_size, y * self.block_size,
                               self.block_size, self.block_size)
        
        if self.main_window.env.current_piece is not None:
            piece = self.main_window.env.current_piece
            pos = self.main_window.env.current_pos
            for y in range(piece.shape[0]):
                for x in range(piece.shape[1]):
                    if piece[y][x]:
                        board_y = pos[0] + y
                        board_x = pos[1] + x
                        if 0 <= board_y < 20 and 0 <= board_x < 10:
                            painter.fillRect(board_x * self.block_size, board_y * self.block_size,
                                          self.block_size, self.block_size, self.colors[2])
                            painter.setPen(QColor(50, 50, 50))
                            painter.drawRect(board_x * self.block_size, board_y * self.block_size,
                                          self.block_size, self.block_size)

class TetrisGameWindow(QMainWindow):
    def __init__(self, model_path: Optional[str] = None):
        """Main window for the Tetris game."""
        super().__init__()
        self.setWindowTitle("Tetris Game")
        self.setMinimumSize(600, 800)
        self.env = TetrisEnv()
        self.agent = None
        if model_path:
            self.agent = TetrisDQN(self.env, TETRIS_DQN_CONFIG)
            self.agent.load(model_path)
        
        self.board_widget = None
        self.setup_ui()
        self.setup_game()
        self.timer = QTimer()
        self.timer.timeout.connect(self.game_loop)
        self.timer.start(1000)
        
        if self.board_widget:
            self.board_widget.setFocus()
    
    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        self.board_widget = TetrisBoardWidget(self)
        main_layout.addWidget(self.board_widget)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.score_label = QLabel("Score: 0")
        self.score_label.setFont(QFont("Arial", 12))
        right_layout.addWidget(self.score_label)
        
        self.level_label = QLabel("Level: 1")
        self.level_label.setFont(QFont("Arial", 12))
        right_layout.addWidget(self.level_label)
        
        self.lines_label = QLabel("Lines: 0")
        self.lines_label.setFont(QFont("Arial", 12))
        right_layout.addWidget(self.lines_label)
        
        new_game_btn = QPushButton("New Game")
        new_game_btn.clicked.connect(self.new_game)
        right_layout.addWidget(new_game_btn)
        
        pause_btn = QPushButton("Pause")
        pause_btn.clicked.connect(self.toggle_pause)
        right_layout.addWidget(pause_btn)
        
        right_layout.addStretch()
        main_layout.addWidget(right_panel)
    
    def setup_game(self):
        """Initialize the game state."""
        self.env.reset()
        self.paused = False
        self.update_labels()
    
    def game_loop(self):
        """Main game loop."""
        if self.env.game_over or self.paused:
            return
        
        if self.agent:
            self.make_ai_move()
        else:
            # Move piece down
            _, reward, done, _, _ = self.env.step(4)  # Simulate gravity
            if done:
                self.show_game_over_dialog()
        
        self.update_labels()
        self.board_widget.update()
    
    def make_ai_move(self):
        """Make AI move."""
        state = self.env.board.copy()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.agent.device)
            q_values = self.agent.policy_net(state_tensor)
            action = q_values.argmax().item()
            _, reward, done, _, _ = self.env.step(action)
            if done:
                self.show_game_over_dialog()
    
    def keyPressEvent(self, event):
        """Handle keyboard input."""
        if self.env.game_over or self.paused:
            return
        
        if event.key() == Qt.Key.Key_Left:
            self.env.step(0)  # Move left
        elif event.key() == Qt.Key.Key_Right:
            self.env.step(1)  # Move right
        elif event.key() == Qt.Key.Key_Up:
            self.env.step(2)  # Rotate
        elif event.key() == Qt.Key.Key_Down:
            self.env.step(4)  # Soft drop
        elif event.key() == Qt.Key.Key_Space:
            # Hard drop
            current_piece = self.env.current_piece.copy()
            current_pos = self.env.current_pos.copy()
            
            # Keep moving down until collision
            while (self.env.current_piece is not None and 
                   np.array_equal(self.env.current_piece, current_piece) and 
                   not self.env.game_over):
                _, _, done, _, _ = self.env.step(4)  # Use soft drop
                if done:
                    self.show_game_over_dialog()
                    break
        
        self.update_labels()
        self.board_widget.update()
    
    def toggle_pause(self):
        """Toggle game pause state."""
        self.paused = not self.paused
        if self.paused:
            self.timer.stop()
        else:
            self.timer.start(1000)
    
    def update_labels(self):
        """Update the score and level labels."""
        self.score_label.setText(f"Score: {self.env.score}")
        self.level_label.setText(f"Level: {self.env.lines_cleared // 10 + 1}")
        self.lines_label.setText(f"Lines: {self.env.lines_cleared}")
    
    def show_game_over_dialog(self):
        """Show a dialog when the game is over."""
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Game Over")
        dialog.setText(f"Game Over!\nFinal Score: {self.env.score}\nLevel: {self.env.lines_cleared // 10 + 1}\nLines: {self.env.lines_cleared}")
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.exec()
    
    def new_game(self):
        """Start a new game."""
        self.setup_game()
        self.timer.start(1000)

def play_tetris_gui(model_path: Optional[str] = None) -> None:
    """Start the Tetris game with GUI interface."""
    app = QApplication(sys.argv)
    window = TetrisGameWindow(model_path)
    window.show()
    sys.exit(app.exec()) 