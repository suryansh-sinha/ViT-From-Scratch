import torch
import torch.nn as nn
import numpy as np
import argparse
from src.vit import ViT
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Vision Transformer (ViT) Setup and Test')
    parser.add_argument('--depth', type=int, default=12, help='Number of attention blocks required')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Probability value for dropout in mlp output')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Probability value for dropout in attention layer')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--weights', type=str, help='Path of weights for trained model if fine-tuning')
    return parser.parse_args()

def setup_model(args):
    model = ViT(depth=args.depth, proj_dropout=args.proj_dropout, attn_dropout=args.attn_dropout)
    
    if args.weights:
        weights_path = Path(args.weights)
        if weights_path.is_file():
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            print(f'Loaded weights from {weights_path}')
        else:
            print(f'Warning: Weights file {weights_path} is not found. Using random initialization.')
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    return model, device

def main():
    args = parse_arguments()
    model, device = setup_model(args)
    
    # Test input
    test_input = torch.randn((16, 3, 384, 384), device=device)
    
    # Inference
    model.eval()
    with torch.inference_mode():
        output = model(test_input)
        
    print(f'Model output shape: {output.shape}')
    print(f'Device used: {device}')
    
if __name__=='__main__':
    main()