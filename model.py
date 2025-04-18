"""
SwinTransformerMultiLabel with Global-Average-Pool Head

This model uses timm to create a Swin Transformer and replaces its head with a global average pooling,
dropout, and a fully connected layer.
"""

import timm
import json
import torch
import torch.nn as nn
from config import MODEL_NAME, DROPOUT_RATE

# ===============================================================
# Model Architecture
# ===============================================================
class AttentionPool2d(nn.Module):
    """
    Simple query-based attention pooling:
      - Projects each spatial location into keys & values.
      - Learns a global query vector to score each location.
      - Outputs a weighted sum over HxW → [B, C].
    """
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, in_channels)) # Learnable query vector of shape [1, C]
        self.to_key = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False) # 1×1 convs for keys & values
        self.to_value = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.scale = in_channels ** -0.5 # Scaling factor for dot‑product attention
    
    def forward(self, x):
        B, C, H, W = x.shape # x: [B, C, H, W]
        # Produce raw keys & values: [B, C, H, W] → [B, C, H*W] then reshape to [B, C, N] and permute to [B, N, C]
        key = self.to_key(x).reshape(B, C, -1).permute(0, 2, 1) # Keys and values: [B, N, C]s
        value = self.to_value(x).reshape(B, C, -1).permute(0, 2, 1)
        query = self.query.expand(B, -1).unsqueeze(1) # Expand single query to one per batch: [1, C] → [B, 1, C]
        attn = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) * self.scale, dim=-1) # Compute scaled dot‑product attention: [B, 1, N]
        output = torch.matmul(attn, value) # Weighted sum of values: [B, 1, C]
        return output.squeeze(1) # Squeeze to [B, C]

class SwinTransformerMultiLabel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # 1) Model backbone without pooling/head
        self.model = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool="")
        # 2) Attention pool + dropout + classifier head
        in_features = self.model.num_features
        self.attn_pool = AttentionPool2d(in_features)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, imgs):
        feat_map = self.model.forward_features(imgs) # Backbone -> feature map [B, H', W', C], downsampled
        feat_map = feat_map.permute(0, 3, 1, 2) # Permute to [B, C, H', W']
        pooled = self.attn_pool(feat_map) # Attention pooling -> [B, C]
        # Head
        features = self.dropout(pooled)
        logits = self.fc(features)  # Pass through a linear fc layer to get one score per class for each example in the batch: [B, num_classes]
        return logits

# ===============================================================
# Classifier Wrapper
# ===============================================================
class Classifier:
    """
    Wrapper for saving, loading, and making predictions.
    
    Args:
        model (nn.Module): The trained model.
        transform (callable): The image preprocessing transformation.
        device (torch.device): The device (CPU or GPU).
        threshold (float): Threshold for converting probabilities to binary predictions.
        labels (list): A list of label names.
    """
    def __init__(self, model, transform, device, labels, thresholds):
        self.model = model.to(device).eval()
        self.transform = transform
        self.device = device
        self.labels = labels 
        self.thresholds = thresholds
        
    def predict(self, img):
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities  = torch.sigmoid(logits).cpu().numpy()[0]  # [K]
        return (probabilities >= self.thresholds).astype(int)

    def save(self, base_filename):
        torch.save(self.model.state_dict(), base_filename + ".pt") # Save model weights
        thresholds_list = self.thresholds.tolist() # Gather thresholds into a JSON‑safe list
        config = { # Build and write the JSON metadata config
            "thresholds": thresholds_list,
            "labels": self.labels
        }
        with open(base_filename + ".json", "w") as file:
            json.dump(config, file, indent=2)
        print(f"Model Weights saved as {base_filename}.pt | Classifier Metadata saved as {base_filename}.json")

    @staticmethod
    def load(json_path, transform, device):
        # Read JSON config w/ inference metadata
        with open(json_path, "r") as f:
            config = json.load(f)
        labels = config["labels"]
        thresholds = config["thresholds"]  
        num_classes = len(labels)

        # Re‑create model architecture
        model = SwinTransformerMultiLabel(num_classes=num_classes, pretrained=True).to(device)
        pt_path = json_path.replace(".json", ".pt") # Get path to weights corresponding to .json inference metadata
        checkpoint = torch.load(pt_path, map_location=device) # Retrieve corresponding weights 
        model.load_state_dict(checkpoint) # Load into architecture
        print(f"Model loaded from {pt_path} and {json_path}.")

        # Wrap in classifier
        classifier = Classifier(model, transform, device, labels=labels, thresholds=thresholds)
        print("Prediction thresholds loaded:", classifier.thresholds)

        return classifier
