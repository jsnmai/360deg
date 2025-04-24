from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

# ===============================================================
# Class for Partitions (Train/Val/Test)
# ===============================================================
class DataPartition(Dataset):
    """
    Initialize the dataset with a DataFrame and transform pipeline.
    Args:
        df (pd.DataFrame): DataFrame with image paths and labels.
        label_columns (list): List of all label names.
        transform: Data augmentation pipeline.
    """
    def __init__(self, df, label_columns, transform=None):
        self.label_columns = label_columns
        self.transform = transform
        self.img_paths = df["image_path"].tolist() # List of image paths
        self.labels = df[label_columns].to_numpy(dtype=np.float32) # 2-D Array of of shape (N_samples, N_labels)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB") # Retrieve image
        if self.transform:                        # Apply transformations to image
            img = self.transform(img)
        label_vector = torch.from_numpy(self.labels[idx]) # Retrieve label vector for the given sample
        return img, label_vector

