import os
import sys
import argparse

sys.path.append("third_party/DiffusionEdge")

from third_party.DiffusionEdge.train_cond_ldm import main as train_main

# 确保模型存储目录
os.makedirs("model", exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train DiffusionEdge on custom dataset')
    parser.add_argument('--cfg', type=str, required=True, help='Path to training config file')
    parser.add_argument('--data_dir', type=str, default="output/diffusion_edge/train", help='Path to training dataset')
    parser.add_argument('--val_dir', type=str, default="output/diffusion_edge/val", help='Path to validation dataset')
    parser.add_argument('--model_path', type=str, default='model/diffusion_edge.pth', help='Path to save trained model')
    return parser.parse_args()

def train(args):
    print(f"Starting training with dataset: {args.data_dir} and validation set: {args.val_dir}")
    sys.argv = ["train_cond_ldm.py", "--cfg", args.cfg]
    train_main()
    print(f"Training complete! Model saved at {args.model_path}")

if __name__ == "__main__":
    args = parse_args()
    train(args)
