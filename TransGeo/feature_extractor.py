import os
import torch
import pandas as pd
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from SSL_Evaluation.Extractor.model import deit_small_distilled_patch16_224
import numpy as np

# Define the MODE flag
MODE = 'query'  # or 'satellite'

# Initialize the model based on the MODE
if MODE == 'query':
    model = deit_small_distilled_patch16_224(pretrained=True, mode='query').to('cuda')
    root_dir = './Query'
    transform = transforms.Compose([
        transforms.Resize((1920, 1080)),
        transforms.ToTensor(),
        # Add other transformations as needed
    ])
    batch_size = 6
else:  # MODE == 'satellite'
    model = deit_small_distilled_patch16_224(pretrained=True, mode='satellite').to('cuda')
    root_dir = './Reference'
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        # Add other transformations as needed
    ])
    batch_size = 12

model.eval()

# Load the dataset with transformations
dataset = ImageFolder(root=root_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

features_list, filename_list = [], []

with torch.no_grad():
    for batch_idx, (images, _) in enumerate(data_loader):
        images = images.to('cuda')

        features = model(images)
        features = features.cpu().numpy()
        features_list.append(features)

        start_idx = batch_idx * data_loader.batch_size
        end_idx = start_idx + len(images)
        filenames = [dataset.samples[i][0] for i in range(start_idx, end_idx)]
        filename_list.extend(filenames)

# Concatenate features into a single numpy array and save
features_array = np.vstack(features_list)
features_file_path = os.path.join(root_dir, 'Features', f'{MODE}_features.npy')
np.save(features_file_path, features_array)

# Save filenames as a CSV file
filenames_file_path = os.path.join(root_dir, 'Features', f'{MODE}_filenames.csv')
pd.DataFrame(filename_list, columns=['filename']).to_csv(filenames_file_path, index=False)
