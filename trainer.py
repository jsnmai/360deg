"""
TrainingMonitor & Trainer

This file handles training, validation, and testing loops with logging for relevant metrics.
It includes support for AMP (automatic mixed precision), early stopping,
and records Validation mAP over epochs.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from torch.amp import autocast

# ===============================================================
# Training Monitor
# ===============================================================
class TrainingMonitor:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_mAPs = []
        self.start=time.time()

    def report_epoch(self, train_loss, val_loss, val_map):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_mAPs.append(val_map)

    def finish(self):
        total_time = time.time()-self.start
        mins = int(total_time // 60)
        secs = int(total_time % 60)
        print(f"Total Training Time: {mins} min {secs} sec")
        return total_time

    def plot(self):
        """
        Plot loss and validation mAP curves.
        """
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(12, 4))
        # Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses,   label="Val Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.grid(True)
        # Validation mAP curve
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.val_mAPs, label="Val mAP")
        plt.title("Validation mAP")
        plt.xlabel("Epoch")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ===============================================================
# Trainer Class with AMP & Early Stopping based on Val Loss
# ===============================================================
class Trainer:
    def __init__(self, model, optimizer, scheduler_cos, scheduler_plateau, criterion, train_loader, val_loader, device, monitor, patience, warmup_epochs, amp_dtype, accumulation_steps):
        self.model = model
        self.optimizer = optimizer
        self.scheduler_cos = scheduler_cos
        self.scheduler_plateau = scheduler_plateau
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.monitor = monitor
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.amp_dtype = amp_dtype
        self.accumulation_steps = accumulation_steps
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.base_lr=optimizer.param_groups[0]['lr'] # store the base LR for warm‑up calculations
        self.best_state = None

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        self.optimizer.zero_grad()
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            with autocast(device_type=self.device.type, dtype=self.amp_dtype): # GPU: forward + loss w/ BF16 Automatic Mixed Precision. Default: FP32 precision
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss = loss / self.accumulation_steps
            loss.backward()# backward pass
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            batch_size = images.size(0)  
            running_loss += loss.item() * batch_size * self.accumulation_steps
            total_samples += batch_size
        if (batch_idx + 1) % self.accumulation_steps != 0: # flush gradients if the last batch didn’t trigger a step
            self.optimizer.step()
            self.optimizer.zero_grad()
        epoch_loss = running_loss / total_samples
        return epoch_loss
    
    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
                batch_size = imgs.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size
                probabilities = torch.sigmoid(logits)
                all_probs.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        val_loss = running_loss / total_samples
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        per_label_AP = [average_precision_score(all_labels[:, i], all_probs[:, i]) for i in range(all_labels.shape[1])]
        val_mAP = float(np.mean(per_label_AP))
        return val_loss, per_label_AP, val_mAP

    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            # warm‑up LR for first few epochs 
            if epoch < self.warmup_epochs:
                warmup_lr = self.base_lr * (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = warmup_lr
            start = time.time()
            train_loss = self.train_epoch()
            val_loss, val_per_label_AP, val_mAP = self.validate_epoch()
            total_time = time.time() - start
            mins = int(total_time // 60)
            secs = int(total_time % 60)
            # Scheduler steps 
            self.scheduler_cos.step()
            self.scheduler_plateau.step(val_loss) 
            # Print epoch summary
            print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val mAP={val_mAP:.4f} ({mins} min {secs} sec)")
            # Early stopping & per‐class AP logging
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.best_state = self.model.state_dict()
                torch.save(self.best_state, "best_model.pt")
                print(f"New best_model.pt saved at epoch {epoch} with val loss: {val_loss:.4f}")
                # Print a little table of per‐class AP
                print("   Validation per-class AP:")
                label_names = self.val_loader.dataset.label_columns
                for name, AP in zip(label_names, val_per_label_AP):
                    print(f"     {name:<15s} {AP:.4f}")
                print(f"   Validation mean AP: {val_mAP:.4f}")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print("Early stopping triggered.")
                    break
            # Record in monitor
            self.monitor.report_epoch(train_loss, val_loss, val_mAP)
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state) # Load best weights
        self.monitor.finish() # Finish training