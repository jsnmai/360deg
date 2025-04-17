"""
Dataset Utilities

This file provides helper functions for loading CSV/JSON files,
performing group-aware stratified splits, and defining the Dataset class,
as well as the data augmentations.
"""
import os
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from config import IMG_WIDTH, device  


# =============================================================================
# JSON Helpers 
# =============================================================================
def get_confidence_score(response_str):
    """
    Extract the confidence score from a response string.
    """
    try:
        return int(response_str.split(" ")[0])
    except Exception:
        return None

def parse_classifications(record, threshold):
    """
    Parse nested classification annotations.
    """
    labels_positive = {}
    projects = record.get("projects", {})
    for project in projects.values():
        for label in project.get("labels", []):
            annotations = label.get("annotations", {})
            for classification in annotations.get("classifications", []):
                label_name = classification.get("name", None)
                positive = False
                checklist = classification.get("checklist_answers", [])
                if checklist:
                    for answer in checklist:
                        score = get_confidence_score(answer.get("name", ""))
                        if score is not None and score <= threshold:
                            positive = True
                            break
                else:
                    val = classification.get("value", "")
                    if val and val[0].isdigit() and int(val[0]) <= threshold:
                        positive = True
                if label_name:
                    labels_positive[label_name] = int(labels_positive.get(label_name, 0) or positive)
    return labels_positive

def get_base_filename(filename):
    """
    Remove suffixes or file extension from the filename.
    """
    for suf in ["_left.jpg", "_right.jpg"]:
        if filename.endswith(suf):
            return filename[:-len(suf)]
    return os.path.splitext(filename)[0]

def create_ndjson_image_path_mapping(ndjson_base_dir):
    """
    Recursively create a mapping from image basename to full path.
    """
    glob_pattern = os.path.join(ndjson_base_dir, "*", "*", "split_jpg", "*.jpg")
    return {os.path.basename(p): p for p in glob.glob(glob_pattern, recursive=True)}


# ===============================================================
# CSV Helper
# ===============================================================
def load_csv_to_df(csv_filepath, images_dir):
    """
    Reads a CSV file into a dataframe and adds 'group_id' and 'image_path' columns.

    Returns:
        pd.DataFrame: A standardized DataFrame with columns:
                      'Filenames', 'group_id', 'image_path', plus any additional label columns.
    """
    df_csv = pd.read_csv(csv_filepath) 
    df_csv["group_id"] = [os.path.splitext(filename)[0] for filename in df_csv["Filenames"]] # Add 'group_id' by removing the file extension.
    df_csv["image_path"] = [os.path.join(images_dir, filename) for filename in df_csv["Filenames"]] # Add 'image_path' by joining the images_dir with the filename.
    return df_csv   


# ===============================================================
# Stratified Split Function
# ===============================================================
def group_stratified_split(df, label_cols, group_col, split_ratio, seed):
    """
    Splits the DataFrame into two subsets in a group-aware manner.
    For each unique group (defined by group_col), this function 
    aggregates the label vectors, and then performs a stratified split 
    based on these aggregated multilabel targets.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        label_cols (list): List of columns representing multilabel targets.
        group_col (str): Column name used to group the data (ie "group_id").
        split_ratio (float): Proportion of groups to use in the "second" split.
        seed (int): Random seed for reproducibility.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrame splits (first split and second split).
    """
    unique_groups = df[group_col].unique()
    aggregated_labels = [] 
    for group in unique_groups:
        group_df = df[df[group_col] == group] # Extract the subset of rows for this group.
        agg_labels = group_df[label_cols].max()  # Use max() across rows for each label column to simulate a logical OR
        aggregated_labels.append(agg_labels)

    unique_groups_array = np.array(unique_groups)
    aggregated_labels_array = np.array(aggregated_labels)
    # Initialize the multilabel stratified shuffle split with the desired test size and random seed.
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=seed)
    
    # Use the splitter to get indices for train and test groups based on the aggregated labels.
    for first_split_idx, second_split_idx in splitter.split(unique_groups_array.reshape(-1, 1), aggregated_labels_array):
        first_groups = unique_groups_array[first_split_idx]
        second_groups = unique_groups_array[second_split_idx]
    
     # Create the final DataFrame splits by selecting rows that belong to each group split.
    df_split_1 = df[df[group_col].isin(first_groups)].reset_index(drop=True)
    df_split_2 = df[df[group_col].isin(second_groups)].reset_index(drop=True)
    return df_split_1, df_split_2


# ===============================================================
# Dataset Class
# ===============================================================
class DataPartition(Dataset):
    def __init__(self, data_frame, label_columns, transform=None):
        """
        Initialize the dataset with a DataFrame and transform pipeline.
        Args:
            data_frame (pd.DataFrame): DataFrame with image paths and labels.
            label_columns (list): List of label column names.
            transform (callable, optional): Data augmentation pipeline.
        """
        self.df = data_frame.copy()
        self.label_columns = label_columns
        self.transform = transform
        self.samples = [(row["image_path"],
                         row[label_columns].values.astype(np.float32))
                        for _, row in self.df.iterrows()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Retrieve the image path and corresponding label vector.
        image_path, label_vector = self.samples[index]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            raise RuntimeError(f"Error loading image from {image_path}: {error}")
        # Apply image transformations.
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label_vector, dtype=torch.float32)


# ===============================================================
# Data Augmentation Pipelines
# ===============================================================
train_transforms = transforms.Compose([
    ##### FEEL FREE TO CHANGE #####
    transforms.Resize((IMG_WIDTH, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
    transforms.RandomApply([transforms.RandomAffine( # Randomly apply
                            degrees=10,                # small rotation: rotate within [-10, 10] degrees
                            translate=(0.05, 0.05),    # small translation: shift up to 5% of the image dimensions
                            scale=(0.95, 1.05))], p=0.5),  # slightly zoom in or out
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
