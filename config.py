"""
Configuration & Global Hyper-Parameters

This file defines the hyper-parameters and settings.
It parses command-line arguments and sets global variables accordingly.
It also sets seeds for reproducibility and configures the computing device.
"""
import os
os.environ["NUMEXPR_MAX_THREADS"] = "16" # (If using GPU, prints clutter if isn't set, irrelevant otherwise)
import argparse
import random
import torch
import numpy as np


# (For using --debug flag) Parse command-line arguments as soon as the module is imported.
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
args, unknown = parser.parse_known_args()

# ===============================================================
# Hyperparameters
# ===============================================================
USE_GPU = True                 # Set if using GPU to train

MODEL_NAME = "swinv2_base_window12to24_192to384"
IMG_WIDTH = 384               
N_EPOCHS = 60                     
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
PATIENCE = 8                   # Early stopping patience
DROPOUT_RATE = 0.5

# --- Scheduler ---
SCHEDULER_T0 = 6               # Number of epochs for the first restart
SCHEDULER_T_MULT = 1           # Multiplier for increasing cycle length after each restart
MIN_LR = 1e-6                  # Minimum learning rate

# --- Partitioning Proportions --- (Ensure total == 1.0)
TRAIN_RATIO = 0.8                   
VAL_RATIO = 0.1                    
TEST_RATIO = 0.1       

# --- Seeds for Reproducibility ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED) # Python built‑in
np.random.seed(RANDOM_SEED) # NumPy
torch.manual_seed(RANDOM_SEED) # PyTorch (CPU)

# --- Per Class OR Global (Single) Threshold for Predictions ---
THRESHOLD_MODE   = 'per_class'   # options: 'per_class', 'global'
GLOBAL_THRESHOLD = 0.5           # only used when THRESHOLD_MODE == 'global'

# If running with --debug flag enabled, override hyperparameters 
if args.debug:
    N_EPOCHS = 2
    BATCH_SIZE = 2

# --- GPU Device Setup -----------------------------------------
if USE_GPU:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

if (USE_GPU and device.type == "cuda"):
    pin_memory = True 
    torch.cuda.manual_seed(RANDOM_SEED) # Seeding for PyTorch (all GPUs)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
else:
    pin_memory = False


if __name__ == "__main__":
    print(f"Using device: {device}")