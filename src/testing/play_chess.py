import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.guis.chess_gui import play_chess_gui

def main():
    parser = argparse.ArgumentParser(description='Play chess against a trained DQN agent')
    parser.add_argument('--model_path', type=str, default='data/checkpoints/chess_model.pth', help='Path to the trained model')
    args = parser.parse_args()
    play_chess_gui(args.model_path)

if __name__ == "__main__":
    main()