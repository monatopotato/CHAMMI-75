import torch
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm

"""
Mean profile over single-cells in the well for DINO4Cells pth files
"""


def aggregate_profiles(plate_filename, profiles_folder, metadata_folder, model):
        feature_size = {'cp-cnn':768, 'dinov1':1920, 'dinov2':1920, 'dinov3':1920, 'OpenPhenom':1920, 'SubCell':4096}
        feature_columns = ['emb_' + str(i) for i in range(feature_size[model])]
        plate_metadata = pd.read_csv(os.path.join(metadata_folder, plate_filename + '.csv' ))
        plate_metadata.drop(columns = ['index', 'Nuclei_Location_Center_X', 'Nuclei_Location_Center_Y', 'Target', 'Metadata_broad_sample', 'val', 'Image_Name' ], inplace = True)
        plate_metadata['batch'] = '2020_11_04_CPJUMP1'
        plate_metadata[['plate', 'well', 'site']] = plate_metadata['Key'].str.split('/', n=3, expand=True)
        plate_metadata.drop(columns = ['Key'], inplace = True)

        if model in ['dinov1', 'dinov2', 'dinov3']:
                features = np.load(os.path.join(profiles_folder, f'{plate_filename}.npy' ))
                print(features.shape)
                features = pd.DataFrame(np.load(os.path.join(profiles_folder, plate_filename + '.npy' )))
                print(features.shape)
        if model == 'SubCell':
                features = pd.DataFrame(np.load(os.path.join(profiles_folder, plate_filename + '.npz' ))['features'])
                print(features.shape)

        assert len(features) == len(plate_metadata)
        profile_data = pd.concat((plate_metadata, features), axis = 1)
        profile_data.rename(columns = dict(zip([i for i in range(feature_size[model])], feature_columns)), inplace = True, errors='raise')
        profile_data = profile_data.groupby(["batch", "plate", "well"])[feature_columns].mean().reset_index()
        plate = plate_filename.split('.')[0]
        os.mkdir(f'./features/aggregated/{model}/{plate}')
        profile_data.to_parquet(f'./features/aggregated/{model}/{plate}/{plate}.parquet')


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Aggregate features to well level profiles")
        parser.add_argument('--profiles', help = 'Path to folder with pth files of plate profiles', required=True)
        parser.add_argument('--metadata', default='./features/metadata', help = 'Path to folder with csv metadata files of plate profiles', required=False)
        parser.add_argument('--model', help = 'With which model features were obtained', required=True)
        args = parser.parse_args()
        os.makedirs(f'./features/aggregated/{args.model}', exist_ok = True)
        plates = [i.split('.')[0] for i in os.listdir(args.profiles)]
        for plate in tqdm(plates):
                aggregate_profiles(plate, args.profiles, args.metadata, args.model)