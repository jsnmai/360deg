"""
Main Driver

To Run:
    python -u main.py | Tee-Object -FilePath metrics.log 

This file loads the data, performs a group-aware stratified split, builds dataloaders,
instantiates the model, optimizer, scheduler, trains the model, 
wraps and saves it in a Classifier.
After training it picks per-class thresholds on validation,
and then evaluates precision/recall/F1 on the test set.
"""

from config import (DEBUG_MODE, USE_GPU, MODEL_NAME, IMG_WIDTH, N_EPOCHS, BATCH_SIZE, LEARNING_RATE, PATIENCE, DROPOUT_RATE,
                    SCHEDULER_T0, SCHEDULER_T_MULT, MIN_LR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED, THRESHOLD_MODE, GLOBAL_PRED_THRESHOLD,
                    device, pin_memory, amp_dtype)
from data_utils import DataPartition, train_transforms, val_transforms, load_csv_to_df, group_stratified_split
from model import SwinTransformerMultiLabel, Classifier
from trainer import TrainingMonitor, Trainer

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support
import warnings # Suppress warnings that currently do not affect execution

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
warnings.filterwarnings("ignore", message="Cannot set number of intraop threads after parallel work has started or after set_num_threads call")

if DEBUG_MODE:
    N_EPOCHS = 2
    BATCH_SIZE = 2

# ===============================================================
# Run/Execute
# ===============================================================
def run():
    ##### FEEL FREE TO CHANGE #####
    CSV_FILE_PATH = 'miml_dataset/miml_labels_1.csv'
    IMG_DIR = 'miml_dataset/images'

    # Print hyperparameters for logging purposes
    params = {
        "DEBUG_MODE": DEBUG_MODE,
        "USE_GPU": USE_GPU,
        "MODEL_NAME": MODEL_NAME,
        "IMG_WIDTH": IMG_WIDTH,
        "N_EPOCHS": N_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "PATIENCE": PATIENCE,
        "DROPOUT_RATE": DROPOUT_RATE,
        "SCHEDULER_T0": SCHEDULER_T0,
        "SCHEDULER_T_MULT": SCHEDULER_T_MULT,
        "MIN_LR": MIN_LR,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "TEST_RATIO": TEST_RATIO,
        "RANDOM_SEED": RANDOM_SEED,
        "THRESHOLD_MODE": THRESHOLD_MODE,
        "GLOBAL_PRED_THRESHOLD": GLOBAL_PRED_THRESHOLD
    }
    print(json.dumps(params, indent=2))

    # Load labels to df
    ##### FEEL FREE TO CHANGE #####
    df = load_csv_to_df(CSV_FILE_PATH, IMG_DIR)
    if DEBUG_MODE:
        df = df.sample(n=200, random_state=RANDOM_SEED).reset_index(drop=True)
    ##### FEEL FREE TO CHANGE #####
    nonlabel_cols = {"external_id", "Filenames", "group_id", "image_path","Problematic", "Extra Notes", "Revisit"}
    label_columns = [col for col in df.columns if col not in nonlabel_cols]
    df[label_columns] = df[label_columns].fillna(0) # Fill NaN entries with 0
    # Split train/val/test partitions
    df_train_and_val, df_test = group_stratified_split(df, label_columns=label_columns, group_col="group_id", split_ratio=TEST_RATIO, seed=RANDOM_SEED)
    relative_val_ratio = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    df_train, df_val = group_stratified_split(df_train_and_val, label_columns=label_columns, group_col="group_id", split_ratio=relative_val_ratio, seed=RANDOM_SEED)
    # Save partitions to .csv
    df_train.to_csv("train_partition.csv", index=False)
    df_val.to_csv("val_partition.csv", index=False)
    df_test.to_csv("test_partition.csv", index=False)
    print("Partitions saved to .csv files.")
    # Load paritions from .csv
    df_train = pd.read_csv("train_partition.csv")
    df_val   = pd.read_csv("val_partition.csv")
    df_test  = pd.read_csv("test_partition.csv")
    # DataLoaders
    train_dataset = DataPartition(df_train, label_columns, transform=train_transforms)
    val_dataset   = DataPartition(df_val,   label_columns, transform=val_transforms)
    test_dataset  = DataPartition(df_test,  label_columns, transform=val_transforms)
    if USE_GPU and torch.cuda.is_available(): # Set num_workers for GPU
        optimal_num_workers = min(8, os.cpu_count() // 2)
    else:
        optimal_num_workers = 0 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=optimal_num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=optimal_num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=optimal_num_workers,pin_memory=pin_memory)
    print(f"Train samples:      {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")
    print(f"Using num_workers: {optimal_num_workers}")
    # Model, optimizer, scheduler
    ##### FEEL FREE TO CHANGE #####
    model = SwinTransformerMultiLabel(num_classes=len(label_columns),pretrained=True).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05, amsgrad=False)
    cos_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=SCHEDULER_T0, T_mult=SCHEDULER_T_MULT, eta_min=MIN_LR)
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-4, cooldown=1, min_lr=MIN_LR)
    # Train 
    monitor = TrainingMonitor()
    ##### ENSURE CHANGES MADE TO MODEL/OPTIMIZER/SCHEDULER ARE REFLECTED HERE TOO #####
    trainer = Trainer(model=model, 
                    optimizer=optimizer,
                    scheduler_cos=cos_scheduler, 
                    scheduler_plateau=plateau_scheduler,
                    criterion=loss_fn,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device, 
                    monitor=monitor,
                    patience=PATIENCE, 
                    warmup_epochs=3, 
                    amp_dtype=amp_dtype,
                    accumulation_steps=2
    )
    trainer.train(N_EPOCHS)
    # Choose Prediction Thresholds on Validation
    if THRESHOLD_MODE == 'per_label': 
        thresholds = find_optimal_thresholds(model, val_loader, device, num_classes=len(label_columns), n_steps=101)
        print("\nOptimal per-class thresholds:", thresholds)
    else: # Single Global Threshold
        thresholds = np.full(len(label_columns), GLOBAL_PRED_THRESHOLD, dtype=float)
    print("Using thresholds:", thresholds)
    # Save in Classifier wrapper w/ Prediction Threshold Settings
    classifier = Classifier(model, val_transforms, device, labels=label_columns, thresholds=thresholds)
    classifier.save("best_classifier")
    # Test set evaluation
    print("\nTest Set performance:")
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probabilities)
            all_labels.append(labels.numpy())
    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    binary_predictions = (all_probs >= thresholds).astype(int)
    # Classification Report on Test Set
    precisions, recalls, f1s, supports = precision_recall_fscore_support(all_labels, binary_predictions, zero_division=0)
    for idx, label in enumerate(label_columns):
        precision = precisions[idx]
        recall = recalls[idx]
        f1 = f1s[idx]
        num_occurrences = supports[idx]
        print(f"{label:<15s} Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, num_occurences={num_occurrences}")

# ===============================================================
# Helper Function for Choosing Prediction Thresholds
# ===============================================================
def find_optimal_thresholds(model, val_loader, device, num_classes, n_steps=101):
    """
    Runs the model on val_loader, collects sigmoid probs & true labels,
    then for each class k finds the threshold tau in [0,1] that maximizes F1.
    Returns: array of thresholds for each class of shape [num_classes].
    """
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    thresholds = np.zeros(num_classes, dtype=float)
    taus = np.linspace(0, 1, n_steps)
    for k in range(num_classes):
        best_f1, best_tau = 0.0, 0.5
        for tau in taus:
            preds_k = (all_probs[:, k] >= tau).astype(int)
            f1 = f1_score(all_labels[:, k], preds_k, zero_division=0)
            if f1 > best_f1:
                best_f1, best_tau = f1, tau
        thresholds[k] = best_tau
    return thresholds


if __name__ == "__main__":
    run()
