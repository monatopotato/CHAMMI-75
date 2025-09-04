'''
Main SimCLR training script with original data loader

'''


import random
import os
from re import match
import sys
import datetime
import time
import math
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torchvision.transforms.v2 import Transform
import sys
sys.path.append("../../")
from dataset.dataset import IterableImageArchive
from dataset import dataset_config
from dataset.dataset_functions import randomize, split_for_workers, get_proc_split
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import distributed_utils
import argparse
import yaml
from multi_channel_vit import get_multi_channel_vit
from torch.nn.parallel import DistributedDataParallel as DDP
from optimizer import get_optimizer
from diffusers.optimization import get_scheduler


class PerImageNormalize(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        # We initialize with num_features=1, but we’ll replace it on-the-fly if needed.
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,             # Temporary placeholder
            affine=False,               # No learnable parameters
            track_running_stats=False,  # Use per-forward stats (no running mean)
            eps=self.eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (N, C, H, W)
        We'll ensure that our instance_norm has the correct number of channels (C).
        """
        # If your input has a dynamic channel size, we need to re-initialize:
        C, _, _ = x.shape
        if self.instance_norm.num_features != C:
            self.instance_norm = nn.InstanceNorm2d(
                num_features=C,
                affine=False,
                track_running_stats=False,
                eps=self.eps
            )

        # Now we can pass x through our InstanceNorm2d layer
        return self.instance_norm(x).to(torch.float16)
    

class SaturationNoiseInjector(nn.Module):
    def __init__(self, low=200, high=255):
        """
        Initialize the SaturationNoiseInjector module.
        
        Parameters:
            low (int): Lower bound for uniform noise values.
            high (int): Upper bound for uniform noise values.
        """
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply high-intensity noise injection to saturated pixels in a single-channel image.
        The function expects the input tensor to have the shape (1, H, W) with pixel intensities in the 0-255 range.

        Process:
          - Convert the input tensor to float32.
          - Generate noise drawn uniformly from [low, high] for each pixel.
          - Create a mask for saturated pixels (where the pixel value equals 255).
          - Zero-out saturated pixels and add the masked noise.

        Parameters:
            x (torch.Tensor): Input tensor of shape (1, H, W).
        
        Returns:
            torch.Tensor: The processed tensor with noise injected.
        """
        # Ensure input is in floating point for correct arithmetic
        x = x.to(torch.float32)
        
        # Since x has one channel, extract the channel as a 2D tensor (H, W)
        channel = x[0]
        
        # Generate noise with values uniformly drawn between self.low and self.high
        noise = torch.empty_like(channel).uniform_(self.low, self.high)
        
        # Create a mask of pixels that are saturated (value == 255)
        mask = (channel == 255).float()
        
        # Apply the mask to the noise to affect only the saturated pixels
        noise_masked = noise * mask
        
        # Remove the saturated pixels by setting them to zero
        channel[channel == 255] = 0
        
        # Add the masked noise to the channel
        channel = channel + noise_masked
        
        # Update the tensor with the modified channel
        x[0] = channel
        
        return x





def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    parser.add_argument('--seed', default=0, type=int, help='Random seed.')

    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")

    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('--data_path', default='/scr/vidit/chammi_train.zip', type=str, help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir', type=str, help='path where to save, empty for no saving')

    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')

    parser.add_argument('--warmup_epoch', default=10, type=int, help='number of warmup epochs')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='number of gradient accumulation steps')

    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Per-GPU batch-size')

    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')

    parser.add_argument('--num_workers', default=6, type=int, help='number of data loading workers per GPU')
    return parser



class SimCLRBatchTransform(object):
    """
    Simple SimCLR transform to apply in your training loop.
    Takes a batch [B, C, H, W] and returns [2*B, C, H, W] with SimCLR ordering.
    """
    
    def __init__(self, image_size=(224, 224), kernel_size=11):
        """
        Args:
            image_size (tuple): Target image size (height, width)
            kernel_size (int): Kernel size for Gaussian blur
        """
        self.image_size = image_size
        self.kernel_size = kernel_size
        
        # Create augmentation pipeline
        self.augmentation_pipeline = v2.Compose([
            v2.RandomResizedCrop(
                size=self.image_size, 
                scale=(0.2, 1.0),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([v2.GaussianBlur(kernel_size=self.kernel_size, sigma=(0.1, 2.0))], p=0.5),
        ])
    
    def __call__(self, batch):
        """
        Apply SimCLR transformations to a batch.
        
        Args:
            batch (torch.Tensor): Input batch [B, C, H, W]
            
        Returns:
            torch.Tensor: Output batch [2*B, C, H, W] ordered as:
                         [img1_view1, img2_view1, ..., img1_view2, img2_view2, ...]
        """
        # Normalize to [0, 1] if input is uint8
        if batch.dtype == torch.uint8:
            batch = batch.float() / 255.0
        elif batch.dtype == torch.float16:
            batch = batch.float()  # Convert float16 to float32
        
        batch_size = batch.shape[0]
        
        # Generate first views
        view1_list = []
        for i in range(batch_size):
            view1 = self.augmentation_pipeline(batch[i])
            view1_list.append(view1)
        
        # Generate second views
        view2_list = []
        for i in range(batch_size):
            view2 = self.augmentation_pipeline(batch[i])
            view2_list.append(view2)
        
        # Stack in SimCLR order: all view1s first, then all view2s
        all_views = view1_list + view2_list
        return torch.stack(all_views, dim=0)

def train_simclr(args):
    distributed_utils.init_distributed_mode(args)
    distributed_utils.fix_random_seeds(args.seed)


    config = dataset_config.DatasetConfig(
                "/scr/vidit/chammi_train.zip", # args.data_path, /scr/data/CHAMMIv2m.zip
                split_fns=[get_proc_split, randomize, split_for_workers],
                num_procs = distributed_utils.get_world_size(), # maybe works? brother needs to check!
                proc = torch.distributed.get_rank(), # This is the global rank generally? Print out later? Look at multinode?
                transform=transforms.Compose([SaturationNoiseInjector(low=200, high=255), PerImageNormalize(), v2.Resize((224,224))]),
                dataset_size="small",
                seed=42,
                use_fp32=True
        )
    
    # Setup the num_epochs as 100
    dataset = IterableImageArchive(config)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=dataset.worker_init_fn, drop_last=True, prefetch_factor=2, pin_memory=True, persistent_workers=True)

    simclr_transform = SimCLRBatchTransform(image_size=(224, 224))
    
    with open("model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)

    model_cfg["in_chans"] = 1
    model = get_multi_channel_vit(**model_cfg).to(args.local_rank)

    # Calculate training steps - CRITICAL for proper scheduler setup
    num_update_steps_per_epoch = math.ceil(len(data_loader) / args.gradient_accumulation_steps)
    num_warmup_steps = num_update_steps_per_epoch * args.warmup_epoch  # 10 epochs of warmup
    total_training_steps = args.epochs * num_update_steps_per_epoch



    ddp_model = DDP(model, device_ids=[args.local_rank])

    channel_ids_list = None  # [0] * b  ## list of channel ids for each image in the batch, used for channelViT simclr
    channel_masks = None
    labels = None
    bag_of_channels_mode = True  ## treat each channel as a separate image

    optimizer = get_optimizer(
        params_to_optimize=[{"params": ddp_model.parameters(), "lr": args.lr}],
        learning_rate=args.lr
    )
    lr_scheduler = get_scheduler(
        name = "cosine",
        optimizer = optimizer,
        num_warmup_steps = num_warmup_steps,
        num_training_steps = total_training_steps
    )


    ddp_model.train()
    
    # Calculate total steps before training starts
    total_steps_per_epoch = len(data_loader) // args.gradient_accumulation_steps
    total_steps = total_steps_per_epoch * args.epochs

    # Setup the learning rate warmup
   # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch + 1}/{args.epochs}")
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(data_loader):
            
            # Apply SimCLR transformations
            simclr_data = simclr_transform(data)
            
            # Forward pass
            output = ddp_model(
                simclr_data,
                channel_ids_list=channel_ids_list,
                channel_masks=channel_masks,
                y=labels,
                bag_of_channels_mode=bag_of_channels_mode,
            )
            
            loss = output["loss"]
            
            # Simplified training step for high batch sizes (no gradient accumulation needed)
            if args.gradient_accumulation_steps == 1:
                # Simple case: update every batch
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Enhanced logging - every 10 steps
                if args.local_rank == 0 and (global_step % 10 == 0 or global_step < 10):
                    current_lr = lr_scheduler.get_last_lr()[0]
                    elapsed_time = time.time() - epoch_start_time
                    steps_in_epoch = batch_idx + 1
                    if steps_in_epoch > 0:
                        time_per_step = elapsed_time / steps_in_epoch
                        eta_epoch = time_per_step * (total_steps_per_epoch - steps_in_epoch)
                        print(f"Epoch {epoch + 1}, Step {global_step}/{total_steps}: "
                            f"Loss = {loss.item():.4f}, LR = {current_lr:.6f}, "
                            f"ETA Epoch: {eta_epoch/60:.1f}min")
            else:
                # Gradient accumulation for memory-limited setups
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        
                        # Enhanced logging - every 10 steps
                        if args.local_rank == 0 and (global_step % 10 == 0 or global_step < 10):
                            current_lr = lr_scheduler.get_last_lr()[0]
                            elapsed_time = time.time() - epoch_start_time
                            if global_step > 0:
                                time_per_step = elapsed_time / (global_step - (epoch * total_steps_per_epoch))
                                eta_epoch = time_per_step * (total_steps_per_epoch - (global_step % total_steps_per_epoch))
                                print(f"Epoch {epoch + 1}, Step {global_step}/{total_steps}: "
                                    f"Loss = {loss.item() * args.gradient_accumulation_steps:.4f}, "
                                    f"LR = {current_lr:.6f}, ETA Epoch: {eta_epoch/60:.1f}min")
            
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            num_batches += 1

    # End of epoch logging with timing
    if args.local_rank == 0:
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches
        current_lr = lr_scheduler.get_last_lr()[0]
        
        # Estimate total training time remaining
        epochs_remaining = args.epochs - (epoch + 1)
        eta_total = (epoch_duration * epochs_remaining) / 60  # in minutes
        
        print(f"Epoch {epoch + 1} completed in {epoch_duration/60:.1f} minutes: "
              f"Average Loss = {avg_epoch_loss:.4f}, LR = {current_lr:.6f}")
        
        if epochs_remaining > 0:
            print(f"ETA for training completion: {eta_total:.1f} minutes ({eta_total/60:.1f} hours)")
        
        print("-" * 50)
    
    # Save checkpoint
    if (epoch + 1) % 1 == 0:  # Save every 1 epochs
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': ddp_model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': avg_epoch_loss,
            'global_step': global_step
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

print("Training completed!")




    # Setup the optimizer


'''
    for epochs in range(100):
        for data in data_loader:

            output = ddp_model(
                data,
                channel_ids_list=channel_ids_list,
                channel_masks=channel_masks,
                y=labels,
                bag_of_channels_mode=bag_of_channels_mode,
            )

            if args.local_rank == 0:
                print("Output keys:", output.keys())
                print("Output shape:", output["output"].shape)
                print("Loss:", output["loss"].item())

            # Here is where the magic happens
            print(data.shape)
            simclr_data = simclr_transform(data)
            print(simclr_data.shape)
            #print(len(data))

            break
'''
        # Save the model now!

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    #Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_simclr(args)

