import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from config.default_config import get_args
from data.loaders import prepare_data_loaders
from utils.training_utils import run_KD

if __name__ == "__main__":
    args = get_args()

    if args.mode == 'train':
        print("Starting Knowledge Distillation training...")
        train_loader, val_loader, test_loader = prepare_data_loaders(args)
        run_KD(train_loader, val_loader, args)
        print("Training finished!")
    else:
        print(f"To run training, execute: python scripts/train_kd.py train [other_args]")
        print(f"Current mode selected is '{args.mode}'. Training logic will not be executed.") 