import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QListWidget, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QFont, QPixmap
import chess
import chess.svg
from cairosvg import svg2png
from io import BytesIO
from PIL import Image
from PIL.ImageQt import ImageQt
import torch
from typing import Optional
from ..environments.chess_env import ChessEnv
from ..algorithms.chess_dqn import ChessDQN
from ..config.chess_dqn_config import CHESS_DQN_CONFIG

class ChessBoardWidget(QWidget):
    """Custom widget for displaying the chess board."""
    
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setMinimumSize(400, 400)
        self.board = chess.Board()
        self.selected_square = None
        self.move_from = None
        self._update_board_display()  # Only update display, not status
    
    def _update_board_display(self):
        """Update only the board display without checking game state."""
        svg_data = chess.svg.board(self.board, size=400)
        png_data = svg2png(bytestring=svg_data.encode('utf-8'))
        image = Image.open(BytesIO(png_data))
        self.board_image = QPixmap.fromImage(ImageQt(image))
        self.update()
    
    def update_board(self):
        """Update the board display and check game state."""
        self._update_board_display()
        self.main_window.update_status()
    
    def paintEvent(self, event):
        """Paint the chess board."""
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.board_image)
        
        # Highlight selected square
        if self.selected_square:
            x = (self.selected_square % 8) * 50
            y = (7 - self.selected_square // 8) * 50
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 0, 100))
            painter.drawRect(x, y, 50, 50)
    
    def mousePressEvent(self, event):
        """Handle mouse clicks for move selection."""
        if event.button() == Qt.MouseButton.LeftButton:
            x = int(event.position().x() // 50)
            y = int(7 - (event.position().y() // 50))
            if 0 <= x <= 7 and 0 <= y <= 7:
                square = y * 8 + x
                
                if self.move_from is None:
                    piece = self.board.piece_at(square)
                    if piece and piece.color == self.board.turn:
                        self.move_from = square
                        self.selected_square = square
                else:
                    move = chess.Move(self.move_from, square)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                        self.main_window.make_ai_move()
                    
                    self.move_from = None
                    self.selected_square = None
                
                self.update_board()

class ChessGameWindow(QMainWindow):
    def __init__(self, model_path: Optional[str] = None):
        """Main window for the chess game."""
        super().__init__()
        self.setWindowTitle("Chess Game")
        self.setMinimumSize(800, 600)
        self.env = ChessEnv()
        self.agent = None
        if model_path:
            self.agent = ChessDQN(self.env, CHESS_DQN_CONFIG)
            self.agent.load(model_path)
        
        self.board_widget = None
        self.setup_ui()
        self.update_status()
    
    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        self.board_widget = ChessBoardWidget(self)
        main_layout.addWidget(self.board_widget)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.status_label = QLabel("White's turn")
        self.status_label.setFont(QFont("Arial", 12))
        right_layout.addWidget(self.status_label)
        
        self.move_history = QListWidget()
        right_layout.addWidget(QLabel("Move History:"))
        right_layout.addWidget(self.move_history)
        
        new_game_btn = QPushButton("New Game")
        new_game_btn.clicked.connect(self.new_game)
        right_layout.addWidget(new_game_btn)
        
        undo_btn = QPushButton("Undo Move")
        undo_btn.clicked.connect(self.undo_move)
        right_layout.addWidget(undo_btn)
        
        right_layout.addStretch()
        main_layout.addWidget(right_panel)
    
    def make_ai_move(self):
        """Make AI move if it's AI's turn."""
        if self.agent and self.board_widget.board.turn == chess.BLACK:
            self.env.board = self.board_widget.board
            self.env._create_move_mapping()
            
            state = self.env._board_to_observation()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                q_values = self.agent.policy_net(state_tensor)
                
                legal_moves = list(self.board_widget.board.legal_moves)
                if legal_moves:
                    legal_actions = [self.env.move_to_idx[move] for move in legal_moves]
                    legal_q_values = q_values[0, legal_actions]
                    best_action_idx = legal_actions[legal_q_values.argmax().item()]
                    move = self.env.idx_to_move[best_action_idx]
                    self.board_widget.board.push(move)
                    self.board_widget.update_board()
                    self.update_move_history()
                    self.update_status()
    
    def update_move_history(self):
        """Update the move history display."""
        move_text = f"{len(self.move_history) + 1}. {self.board_widget.board.peek().uci()}"
        self.move_history.addItem(move_text)
        self.update_status()
    
    def update_status(self):
        """Update the game status display."""
        if self.board_widget.board.is_checkmate():
            winner = "White" if self.board_widget.board.turn == chess.BLACK else "Black"
            self.status_label.setText(f"Checkmate! {winner} wins!")
            self.show_game_over_dialog(f"Checkmate! {winner} wins!")
        elif self.board_widget.board.is_stalemate():
            self.status_label.setText("Stalemate! Game Over")
            self.show_game_over_dialog("Game ended in stalemate!")
        elif self.board_widget.board.is_insufficient_material():
            self.status_label.setText("Draw! Insufficient material")
            self.show_game_over_dialog("Game ended in draw - insufficient material!")
        else:
            turn = "White" if self.board_widget.board.turn == chess.WHITE else "Black"
            if self.board_widget.board.is_check():
                self.status_label.setText(f"{turn}'s turn - CHECK!")
            else:
                self.status_label.setText(f"{turn}'s turn")
    
    def show_game_over_dialog(self, message: str):
        """Show a dialog when the game is over."""
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Game Over")
        dialog.setText(message)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.exec()
    
    def new_game(self):
        """Start a new game."""
        self.board_widget.board = chess.Board()
        self.board_widget.update_board()
        self.move_history.clear()
        self.update_status()
    
    def undo_move(self):
        """Undo the last move."""
        if len(self.board_widget.board.move_stack) > 0:
            self.board_widget.board.pop()
            if self.agent and len(self.board_widget.board.move_stack) > 0:
                self.board_widget.board.pop()
            self.board_widget.update_board()
            self.move_history.takeItem(self.move_history.count() - 1)
            self.update_status()

def play_chess_gui(model_path: Optional[str] = None) -> None:
    """Start the chess game with GUI interface."""
    app = QApplication(sys.argv)
    window = ChessGameWindow(model_path)
    window.show()
    sys.exit(app.exec()) 