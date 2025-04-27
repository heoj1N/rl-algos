"""Script to play chess against the trained AI."""

from src.gui.chess_game import play_chess

if __name__ == "__main__":
    # Use the trained model from the data directory
    model_path = "data/chess_dqn_model.pth"
    
    play_chess(model_path) 