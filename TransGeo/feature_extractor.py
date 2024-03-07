import os
import torch
import pandas as pd
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from SSL_Evaluation.Extractor.model import deit_small_distilled_patch16_224
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Feature Extraction Script')
parser.add_argument('--mode', type=str, required=True, choices=['query', 'satellite'],
                    help='Operational mode: query or satellite.')
parser.add_argument('--query_dir', type=str, default='/path/to/query/dataset',
                    help='Directory for query dataset.')
parser.add_argument('--satellite_dir', type=str, default='/path/to/satellite/dataset',
                    help='Directory for satellite dataset.')
args = parser.parse_args()

# Initialize the model based on the mode
if args.mode == 'query':
    model = deit_small_distilled_patch16_224(pretrained=True, mode='query').to('cuda')
    root_dir = args.query_dir
    transform = transforms.Compose([
        transforms.Resize((1920, 1080)),
        transforms.ToTensor(),
    ])
    batch_size = 5
else:  # args.mode == 'satellite'
    model = deit_small_distilled_patch16_224(pretrained=True, mode='satellite').to('cuda')
    root_dir = args.satellite_dir
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    batch_size = 12
    use_cropping = False  # Optional: Set to True if cropping is needed in satellite mode

model.eval()

# Load the dataset with transformations
dataset = ImageFolder(root=root_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

features_list, filename_list = [], []

with torch.no_grad():
    for batch_idx, (images, _) in enumerate(data_loader):
        images = images.to('cuda')

        # Optional cropping logic for satellite mode
        if args.mode == 'satellite' and use_cropping:
            # Implement cropping logic here
            pass

        features = model(images)
        features = features.cpu().numpy()
        features_list.append(features)

        start_idx = batch_idx * data_loader.batch_size
        end_idx = start_idx + len(images)
        filenames = [dataset.samples[i][0] for i in range(start_idx, end_idx)]
        filename_list.extend(filenames)

# Concatenate features into a single numpy array and save
features_array = np.vstack(features_list)
features_file_path = os.path.join(root_dir, 'Features', f'{args.mode}_features.npy')
np.save(features_file_path, features_array)

# Save filenames as a CSV file
filenames_file_path = os.path.join(root_dir, 'Features', f'{args.mode}_filenames.csv')
pd.DataFrame(filename_list, columns=['filename']).to_csv(filenames_file_path, index=False)
