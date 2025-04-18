"""
Configuration & Global Hyper-Parameters

This file defines the hyper-parameters and settings.
It parses command-line arguments and sets global variables accordingly.
It also sets seeds for reproducibility and configures the computing device.
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "16" # (If using GPU, prints clutter if isn't set, irrelevant otherwise)
import random
import torch
import numpy as np

# ===============================================================
# Hyperparameters
# ===============================================================
# Training / Model
DEBUG_MODE = True
USE_GPU = True                 # Set if using GPU to train
MODEL_NAME = "swinv2_base_window12to24_192to384"
IMG_WIDTH = 384               
N_EPOCHS = 60                     
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
PATIENCE = 8                   # Early stopping patience
DROPOUT_RATE = 0.5
# Scheduler
SCHEDULER_T0 = 6               # Number of epochs for the first restart
SCHEDULER_T_MULT = 1           # Multiplier for increasing cycle length after each restart
MIN_LR = 1e-6                  # Minimum learning rate
# Partitioning Proportions (Ensure total == 1.0)
TRAIN_RATIO = 0.8                   
VAL_RATIO = 0.1                    
TEST_RATIO = 0.1       
RANDOM_SEED = 42
# Per Label OR Global (Single) Threshold for Predictions 
THRESHOLD_MODE   = 'per_label'   # options: 'per_label', 'global'
GLOBAL_PRED_THRESHOLD = 0.5      # only used when THRESHOLD_MODE == 'global'

# Reproducibility 
random.seed(RANDOM_SEED) # Python built‑in
np.random.seed(RANDOM_SEED) # NumPy
torch.manual_seed(RANDOM_SEED) # PyTorch (CPU)

# Device setup
if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    pin_memory = True
    amp_dtype = torch.bfloat16
else:
    device = torch.device("cpu")
    pin_memory = False
    amp_dtype = torch.float32


if __name__ == "__main__":
    print(f"Using device: {device}")