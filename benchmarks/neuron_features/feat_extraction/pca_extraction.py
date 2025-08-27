import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from accelerate import Accelerator
from sklearn.decomposition import PCA

from dataloader import CellDataset, ToTensorNormalize
from vision_transformer import vit_small

class ViTClass():
    def __init__(self, device):
        self.device = device
        self.model = vit_small()
        remove_prefixes = ["module.backbone.", "module.", "module.head."]
        student_model = torch.load("/scr/vidit/Models/DINO_CHAMMI-75_LARGE/checkpoint.pth")['student']
        cleaned_state_dict = {}
        for k, v in student_model.items():
            new_key = k
            for prefix in remove_prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            if not new_key.startswith("head.mlp") and not new_key.startswith("head.last_layer"):
                cleaned_state_dict[new_key] = v
        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)

    def get_model(self):
        return self.model

def extract_embeddings(dataloader, model, accelerator):
    embeddings = []
    gene_names = []
    vit_model = model.get_model()
    for data in tqdm(dataloader, desc="Extracting embeddings"):
        image_tensor = data['image_tensor'][0]  # [N_CH, 64, 64]
        sample_embeddings = []
        for channel in image_tensor:
            channel_data = channel.unsqueeze(0).unsqueeze(0).to(accelerator.device)
            with torch.no_grad():
                output = vit_model.forward_features(channel_data)
                embedding = output["x_norm_clstoken"].cpu().detach().numpy().flatten()
                sample_embeddings.append(embedding)
        # After collecting all channel embeddings for this sample, stack them
        sample_embeddings = np.stack(sample_embeddings)  # shape: (14, 384)
        # Apply PCA to get first 20 columns
        pca = PCA(n_components=20)
        sample_embeddings_pca = pca.fit_transform(sample_embeddings)  # shape: (14, 20)
        print(sample_embeddings_pca.shape)
        embeddings.append(sample_embeddings_pca)
        gene_names.append(data['gene_name'][0] if isinstance(data['gene_name'], (list, tuple)) else data['gene_name'])
    embeddings = np.array(embeddings)
    # Flatten embeddings to 2D for PCA: (num_samples * num_channels, embedding_dim)
    num_samples, num_channels, embedding_dim = embeddings.shape
    embeddings_2d = embeddings.reshape(-1, embedding_dim)
    pca = PCA(n_components=20)
    embeddings_pca = pca.fit_transform(embeddings_2d)
    # Reshape back to (num_samples, num_channels, n_components)
    embeddings_pca = embeddings_pca.reshape(num_samples, num_channels, -1)
    return embeddings_pca, gene_names

def main():
    accelerator = Accelerator()
    train_dataset = CellDataset(
        datadir='/scr/vidit/neural_features/input_data',
        mode='train',
        transform=ToTensorNormalize(),
        mask_flag=False
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=0
    )
    train_dataloader = accelerator.prepare(train_dataloader)
    model = ViTClass(device=accelerator.device)
    embeddings, gene_names = extract_embeddings(train_dataloader, model, accelerator)

if __name__ == "__main__":
    main()
