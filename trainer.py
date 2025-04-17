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
from torch.amp import autocast, GradScaler


# ===============================================================
# Training Monitor
# ===============================================================
class TrainingMonitor:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_mAPs = []
        self.start_time = time.time()

    def report_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, val_map):
        """
        Record metrics for the current epoch.
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.val_mAPs.append(val_map)

    def finish(self):
        total_time = time.time() - self.start_time
        mins = int(total_time // 60)
        secs = int(total_time % 60)
        print(f"Total Training Time: {mins} min {secs} sec")
        return total_time

    def plot(self):
        """
        Plot loss, accuracy, and validation mAP curves.
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
        # Accuracy curves
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_accuracies, label="Train Acc")
        plt.plot(epochs, self.val_accuracies,   label="Val Acc")
        plt.title("Accuracy")
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
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cos_scheduler,
        plateau_scheduler,
        criterion,
        train_loader,
        val_loader,
        device,
        monitor: TrainingMonitor,
        patience: int,
        warmup_epochs: int = 3,
        accumulation_steps: int = 1
    ):
        self.model = model
        self.optimizer = optimizer
        self.cos_scheduler = cos_scheduler
        self.plateau_scheduler = plateau_scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.monitor = monitor
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.accumulation_steps = accumulation_steps

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_state = None
        self.scaler = GradScaler()

        # store the base LR for warm‑up calculations
        self.base_lr = optimizer.param_groups[0]['lr']

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()
        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            with autocast(device_type=self.device.type):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss += loss.item() * images.size(0) * self.accumulation_steps
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                running_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(logits)

                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()

                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss = running_loss / total
        val_acc = correct / total
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)

        # Per‐class AP and mean AP
        per_class_AP = [
            average_precision_score(all_labels[:, i], all_probs[:, i])
            for i in range(all_labels.shape[1])
        ]
        val_mAP = float(np.mean(per_class_AP))

        return val_loss, val_acc, per_class_AP, val_mAP

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            # — warm‑up LR for first few epochs —
            if epoch < self.warmup_epochs:
                warmup_lr = self.base_lr * (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = warmup_lr

            start = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_per_class_AP, val_mAP = self.validate_epoch()
            elapsed = time.time() - start
            mins, secs = divmod(int(elapsed), 60)

            # — scheduler steps —
            self.cos_scheduler.step()
            self.plateau_scheduler.step(val_loss)

            # — epoch summary —
            print(
                f"\nEpoch {epoch+1}: "
                f"Train Loss={train_loss:.4f} | "
                f"Val Loss={val_loss:.4f} | "
                f"Val mAP={val_mAP:.4f} "
                f"({mins} min {secs} sec)"
            )

            # — early stopping & per‐class AP logging —
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = self.model.state_dict()
                torch.save(self.best_state, "best_model.pt")
                self.epochs_no_improve = 0
                print(f"New best_model.pt saved at epoch {epoch+1} with val loss: {val_loss:.4f}")

                # Print a little table of per‐class AP
                label_names = self.val_loader.dataset.label_columns
                print("   Validation per-class AP:")
                for name, ap in zip(label_names, val_per_class_AP):
                    print(f"     {name:<15s} {ap:.4f}")
                print(f"   Validation mean AP: {val_mAP:.4f}")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print("Early stopping triggered.")
                    break

            # record in monitor
            self.monitor.report_epoch(epoch+1, train_loss, train_acc, 
                                        val_loss, val_acc, val_mAP)

        # load best weights & finish
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.monitor.finish()
