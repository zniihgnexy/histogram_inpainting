import os
import sys
import argparse

sys.path.append("third_party/DiffusionEdge")

from third_party.DiffusionEdge.demo import main as test_main

def parse_args():
    parser = argparse.ArgumentParser(description='Test DiffusionEdge on custom dataset')
    parser.add_argument('--input_dir', type=str, default="output/diffusion_edge/val/image", help='Path to validation dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--output_dir', type=str, default='results/', help='Path to save output images')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    return parser.parse_args()

def test(args):
    print(f"Starting inference on validation set: {args.input_dir}")
    sys.argv = ["demo.py", "--input_dir", args.input_dir, "--pre_weight", args.model_path, "--out_dir", args.output_dir, "--bs", str(args.batch_size)]
    test_main()
    print(f"Inference complete! Results saved in {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    test(args)
