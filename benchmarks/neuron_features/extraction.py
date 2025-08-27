import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import argparse
from accelerate import Accelerator

from dataloader import CellDataset, ToTensorNormalize
import sys
sys.path.append("../")
from models import ViTClass, get_model
import os

def extract_embeddings(dataloader, model, accelerator):
    embeddings = []
    
    # Debug: Print how many samples each GPU will process
    print(f"GPU {accelerator.local_process_index} will process {len(dataloader)} samples")
    
    # Only show progress on main process to avoid multiple progress bars
    if accelerator.is_main_process:
        iterator = tqdm(dataloader, desc=f"Extracting embeddings")
    else:
        iterator = dataloader
    
    for data in iterator:
        image_tensor = data['image_tensor']  # Remove batch dimension: [N_CH, 64, 64]
        
        image_embedding = model(image_tensor.to(accelerator.device))  # [N_CH, D]
        metadata = {k: data[k] for k in data if k != 'image_tensor'}
        embeddings.append({'embedding': image_embedding, 'metadata': metadata})
    
    return embeddings

def main():
    parser = argparse.ArgumentParser(description='Extract features using VIT or subcell model')
    parser.add_argument('--model', type=str, choices=['vit', 'subcell', 'dinov2', 'openphenom', 'mae'], default='vit',
                        help='Model to use for feature extraction (default: vit)')
    parser.add_argument('--config_path', type=str, default="/mnt/cephfs/mir/jcaicedo/morphem/dataset/models/subcell_models/all_channels_ViT-ProtS-Pool.yaml",
                        help='Path to config file for subcell model (required when using subcell)')
    parser.add_argument('--image_folder', type=str, default="/scr/data/cell_crops",
                        help='Path to image folder', required=True)
    parser.add_argument('--output_folder', type=str, default="/scr/data/HPA_features",
                        help='Output folder for features', required=True)
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--model_path', type=str, default="", help = "Path to where the model is located")
    parser.add_argument('--model_size', type=str, choices=['small', 'base'], default='small')
    args = parser.parse_args()

    accelerator = Accelerator()
    
    # Create datasets first
    train_dataset = CellDataset(
        datadir=args.image_folder,
        mode='train',
        mask_flag=True
    )
    test_dataset = CellDataset(
        datadir=args.image_folder,
        mode='test',
        mask_flag=True
    )
    print("Masking parm is on!")

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers  # Start with 0 for debugging, increase later if needed
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Prepare dataloaders for multi-GPU distribution
    train_dataloader, test_dataloader = accelerator.prepare(train_dataloader, test_dataloader)

    # Initialize model
    model_instance = get_model(model_name=args.model, device=accelerator.device, model_path=args.model_path, model_size=args.model_size)
    model_instance.to(accelerator.device)

    # Extract embeddings
    train_embeddings = extract_embeddings(train_dataloader, model_instance, accelerator)
    test_embeddings = extract_embeddings(test_dataloader, model_instance, accelerator)

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)
    train_path = os.path.join(args.output_folder, "train_embeddings.pkl")
    test_path = os.path.join(args.output_folder, "test_embeddings.pkl")

    # Gather all embeddings from all processes
    if accelerator.num_processes > 1:
        # Gather embeddings from all GPUs
        all_train_embeddings = accelerator.gather_for_metrics(train_embeddings)
        all_test_embeddings = accelerator.gather_for_metrics(test_embeddings)
        
        # Only save on main process to avoid duplicate files
        if accelerator.is_main_process:
            print(f"Gathered {len(all_train_embeddings)} training embeddings from all GPUs")
            print(f"Gathered {len(all_test_embeddings)} test embeddings from all GPUs")
            
            with open(train_path, "wb") as f:
                pickle.dump(all_train_embeddings, f)
            with open(test_path, "wb") as f:
                pickle.dump(all_test_embeddings, f)
    else:
        # Single GPU case
        with open(train_path, "wb") as f:
            pickle.dump(train_embeddings, f)
        with open(test_path, "wb") as f:
            pickle.dump(test_embeddings, f)

    if accelerator.is_main_process:
        print("Embedding extraction complete!")

if __name__ == "__main__":
    main()