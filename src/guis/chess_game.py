import chess
import chess.svg
import torch
from typing import Optional
from ..environments.chess_env import ChessEnv
from ..algorithms.chess_dqn import ChessDQN
from ..config.chess_dqn_config import CHESS_DQN_CONFIG

class ChessGame:
    def __init__(self, model_path: Optional[str] = None):
        """Simple chess game interface using python-chess."""    
        self.env = ChessEnv()
        self.board = self.env.board
        self.agent = None
        if model_path:
            self.agent = ChessDQN(self.env, CHESS_DQN_CONFIG)
            self.agent.load(model_path)
    
    def _get_ai_move(self) -> Optional[chess.Move]:
        """Get move from AI agent."""
        if self.agent and self.board.turn == chess.BLACK:
            self.env._create_move_mapping()
            state = self.env._board_to_observation()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                q_values = self.agent.policy_net(state_tensor)
                # Get all legal moves
                legal_moves = list(self.board.legal_moves)
                if not legal_moves:
                    return None
                # Get Q-values for legal moves only
                legal_actions = [self.env.move_to_idx[move] for move in legal_moves]
                legal_q_values = q_values[0, legal_actions]
                # Choose move with highest Q-value
                best_action_idx = legal_actions[legal_q_values.argmax().item()]
                return self.env.idx_to_move[best_action_idx]
        return None
    
    def _print_board(self) -> None:
        """Print the current board state with file and rank labels."""
        board_str = str(self.board)
        board_lines = board_str.split('\n')
        file_labels = '  a b c d e f g h  '
        print(file_labels)
        for i, line in enumerate(board_lines):
            rank = 8 - i
            print(f"{rank} {line} {rank}")
        print(file_labels)
        print()
    
    def _get_user_move(self) -> chess.Move:
        """Get a move from the user."""
        while True:
            try:
                move_str = input("Enter your move (e.g., e2e4): ")
                move = chess.Move.from_uci(move_str)
                if move in self.board.legal_moves:
                    return move
                print("Invalid move! Try again.")
            except ValueError:
                print("Invalid move format! Use format like 'e2e4'.")
    
    def run(self) -> None:
        """Run the chess game."""
        print("Welcome to Chess!")
        print("You are playing as White. Enter moves in the format 'e2e4'.")
        print("Type 'quit' to exit the game.")
        
        while not self.board.is_game_over():
            self._print_board()
            
            if self.board.turn == chess.WHITE:
                move = self._get_user_move()
                self.board.push(move)
            else:
                if self.agent:
                    print("AI is thinking...")
                    move = self._get_ai_move()
                    if move:
                        print(f"AI plays: {move.uci()}")
                        self.board.push(move)
                    else:
                        print("AI couldn't find a move!")
                        break
                else:
                    move = self._get_user_move()
                    self.board.push(move)
        
        self._print_board()
        if self.board.is_checkmate():
            print("Checkmate! " + ("White" if self.board.turn == chess.BLACK else "Black") + " wins!")
        elif self.board.is_stalemate():
            print("Stalemate!")
        elif self.board.is_insufficient_material():
            print("Draw by insufficient material!")
        else:
            print("Game over!")

def play_chess(model_path: Optional[str] = None) -> None:
    """Start a chess game against the AI."""
    game = ChessGame(model_path)
    game.run() 