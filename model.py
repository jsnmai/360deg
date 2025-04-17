"""
SwinTransformerMultiLabel with Global-Average-Pool Head

This model uses timm to create a Swin Transformer and replaces its head with a global average pooling,
dropout, and a fully connected layer.
"""
import timm
import json
import torch
import torch.nn as nn
import numpy as np
from config import MODEL_NAME, DROPOUT_RATE, GLOBAL_THRESHOLD


# ===============================================================
# Model Architecture
# ===============================================================
class AttentionPool2d(nn.Module):
    """
    Simple query-based attention pooling:
      - Projects each spatial location into keys & values.
      - Learns a global query vector to score each location.
      - Outputs a weighted sum over HxW -> [B, C].
    """
    def __init__(self, in_channels):
        super().__init__()
        # Query vector of shape [1, C]
        self.query = nn.Parameter(torch.randn(1, in_channels))
        # 1×1 convs for keys & values
        self.to_k = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.scale = in_channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape # x: [B, C, H, W]
        # Keys and values: [B, N, C]
        k = self.to_k(x).view(B, C, -1).permute(0, 2, 1)
        v = self.to_v(x).view(B, C, -1).permute(0, 2, 1)
        # Expand query to [B, 1, C]
        q = self.query.expand(B, -1).unsqueeze(1)
        # Attention scores [B, 1, N]
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        # Weighted sum of values -> [B, 1, C]
        out = torch.matmul(attn, v)
        # Squeeze to [B, C]
        return out.squeeze(1)


class SwinTransformerMultiLabel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # 1) Model backbone without pooling/head
        self.model = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool=""
        )
        in_features = self.model.num_features

        # 2) Attention pool + dropout + classifier head
        self.attn_pool = AttentionPool2d(in_features)
        self.dropout   = nn.Dropout(DROPOUT_RATE)
        self.fc        = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat_map = self.model.forward_features(x) # Extract feature map [B, H, W, C]
        feat_map = feat_map.permute(0, 3, 1, 2).contiguous() # Permute to [B, C, H, W]
        pooled = self.attn_pool(feat_map) # Attention pooling -> [B, C]
        # Head
        dropped  = self.dropout(pooled)
        logits   = self.fc(dropped)  # [B, num_classes]
        return logits


# ===============================================================
# Classifier Wrapper
# ===============================================================
class Classifier:
    def __init__(self, model, transform, device, labels, thresholds):
        """
        Wrapper for saving, loading, and making predictions.
        
        Args:
            model (nn.Module): The trained model.
            transform (callable): The image preprocessing transformation.
            device (torch.device): The device (CPU or GPU).
            threshold (float): Threshold for converting probabilities to binary predictions.
            labels (list): A list of label names.
        """
        self.model = model.to(device).eval()
        self.transform = transform
        self.device = device
        self.thresholds = thresholds
        self.labels = labels or []

    def predict(self, pil_img):
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.sigmoid(logits).cpu().numpy()[0]  # [K]
        return (probs >= self.thresholds).astype(float)

    def save(self, base_name):
        torch.save(self.model.state_dict(), base_name + ".pt") # Save model weights
        thresh_list = self.thresholds.tolist() # Gather thresholds into a JSON‑safe list
        config = { # Build and write the JSON metadata config
            "thresholds": thresh_list,
            "labels": self.labels
        }
        with open(base_name + ".json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"Model Weights saved as {base_name}.pt | Classifier Metadata saved as {base_name}.json")

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
