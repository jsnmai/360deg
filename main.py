"""
Main Driver

To Run Full Training:
    python -u main.py | Tee-Object -FilePath metrics.log 

To Run Small-scale Test         
    python -u main.py --debug | Tee-Object -FilePath metrics.log  

This file loads the data, performs a group-aware stratified split, builds dataloaders,
instantiates the model, optimizer, scheduler, trains the model, 
wraps and saves it in a Classifier.
After training it picks per-class thresholds on validation,
and then evaluates precision/recall/F₁ on the test set.
"""
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support

from config import args
from config import (USE_GPU, MODEL_NAME, IMG_WIDTH, N_EPOCHS, BATCH_SIZE, LEARNING_RATE, PATIENCE, DROPOUT_RATE,
                    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED, THRESHOLD_MODE, GLOBAL_THRESHOLD,
                    device, pin_memory, SCHEDULER_T0, SCHEDULER_T_MULT, MIN_LR)
import dataset as ds
from dataset import train_transforms, val_transforms
from model import SwinTransformerMultiLabel, Classifier
from trainer import TrainingMonitor, Trainer

import warnings # Suppress warnings that currently do not affect execution
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
warnings.filterwarnings("ignore", message="Cannot set number of intraop threads after parallel work has started or after set_num_threads call")


# ===============================================================
# Run/Execute
# ===============================================================
def run(debug=False):
    ##### FEEL FREE TO CHANGE #####
    CSV_FILE_PATH = 'miml_dataset/miml_labels_1.csv'
    IMG_DIR = 'miml_dataset/images'

    params = { # Print hyperparameters for logging purposes
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
        "USE_GPU": USE_GPU,
        "THRESHOLD_MODE": THRESHOLD_MODE
    }
    print(json.dumps(params, indent=2))

    # --- Data Load & Split ---
    ##### FEEL FREE TO CHANGE #####
    df = ds.load_csv_to_df(CSV_FILE_PATH, IMG_DIR)

    if debug:
        df = df.sample(n=200, random_state=RANDOM_SEED).reset_index(drop=True)

    ##### FEEL FREE TO CHANGE #####
    nonlabel_cols = {"external_id", "Filenames", "group_id", "image_path",
                     "Problematic", "Extra Notes", "Revisit"}

    label_columns = [c for c in df.columns if c not in nonlabel_cols]
    df[label_columns] = df[label_columns].fillna(0)

    df_train_and_val, df_test = ds.group_stratified_split(df, label_columns, "group_id", TEST_RATIO, RANDOM_SEED)
    rel_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    df_train, df_val = ds.group_stratified_split(df_train_and_val, label_columns, "group_id", rel_val, RANDOM_SEED)

    df_train.to_csv("train_partitions.csv", index=False)
    df_val.to_csv("val_partitions.csv", index=False)
    df_test.to_csv("test_partitions.csv", index=False)
    print("Partition CSV files saved.")

    df_train = pd.read_csv("train_partitions.csv")
    df_val   = pd.read_csv("val_partitions.csv")
    df_test  = pd.read_csv("test_partitions.csv")

    # --- Dataloaders ---
    if USE_GPU:
        if debug:
            optimal_workers = 2
        else:
            optimal_workers = min(8, os.cpu_count() // 2)
    else:
        optimal_workers = 0
    train_dataset = ds.DataPartition(df_train, label_columns, transform=train_transforms)
    val_dataset   = ds.DataPartition(df_val,   label_columns, transform=val_transforms)
    test_dataset  = ds.DataPartition(df_test,  label_columns, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=optimal_workers,
                              pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=optimal_workers,
                              pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=optimal_workers,
                              pin_memory=pin_memory)
    print(f"Train samples:      {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")
    print(f"Using num_workers: {optimal_workers}")

    # --- Model / Optimizer / Scheduler ---
    ##### FEEL FREE TO CHANGE #####
    model     = SwinTransformerMultiLabel(num_classes=len(label_columns), pretrained=True).to(device)
    loss_fn   = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05, amsgrad=False)
    ca_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=SCHEDULER_T0, T_mult=SCHEDULER_T_MULT, eta_min=MIN_LR)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-4, cooldown=1, min_lr=MIN_LR)

    # --- Train ---
    monitor = TrainingMonitor()
    ##### ENSURE MATCHES ABOVE #####
    trainer = Trainer(model=model, 
                    optimizer=optimizer,
                    cos_scheduler=ca_scheduler, 
                    plateau_scheduler=plateau_scheduler,
                    criterion=loss_fn,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device, 
                    monitor=monitor,
                    patience=PATIENCE, 
                    warmup_epochs=3, 
                    accumulation_steps=2
    )

    trainer.train(N_EPOCHS)

    # --- Choose Prediction Thresholds ---
    if THRESHOLD_MODE == 'per_class': # Pick Thresholds on Validation
        thresholds = search_optimal_thresholds(model, val_loader, device, num_classes=len(label_columns), n_steps=101)
        print("\nOptimal per-class thresholds:", thresholds)
    else: # Single Global Threshold
        thresholds = np.full(len(label_columns), GLOBAL_THRESHOLD, dtype=float)
    print("Using thresholds:", thresholds)

    # --- Save Classifier ---
    classifier = Classifier(model, val_transforms, device, labels=label_columns, thresholds=thresholds)
    classifier.save("best_classifier")

    # --- Final Test Set Evaluation ---
    print("\nTest Set performance:")
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    preds = (all_probs >= thresholds).astype(int)

    prec, rec, f1, num_occurences = precision_recall_fscore_support(
        all_labels, preds, zero_division=0
    )
    for i, label in enumerate(label_columns):
        print(f"{label:<15s} Precision={prec[i]:.4f}, Recall={rec[i]:.4f}, "
              f"F1={f1[i]:.4f}, num_occurences={num_occurences[i]}")


# ===============================================================
# Helper Function for Choosing Prediction Thresholds
# ===============================================================
def search_optimal_thresholds(model, val_loader, device, num_classes, n_steps=101):
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
    run(debug=args.debug)
