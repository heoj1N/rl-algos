import argparse
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from guis.tetris_gui import play_tetris_gui

def main():
    parser = argparse.ArgumentParser(description='Play Tetris against a trained AI model')
    parser.add_argument('--model_path', type=str, help='Path to the trained model checkpoint')
    args = parser.parse_args()
    
    play_tetris_gui(args.model_path)

if __name__ == '__main__':
    main() 