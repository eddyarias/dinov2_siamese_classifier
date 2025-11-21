import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        # Distancias Euclidianas
        d_pos = torch.norm(anchor - positive, p=2, dim=-1)
        d_neg = torch.norm(anchor - negative, p=2, dim=-1)
        losses = torch.relu(d_pos - d_neg + self.margin)
        loss = losses.mean()
        return loss, d_pos.mean().item(), d_neg.mean().item()

__all__ = ["TripletLoss"]
