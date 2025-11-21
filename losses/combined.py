import torch.nn as nn
from .triplet import TripletLoss

class CombinedLoss(nn.Module):
    def __init__(self, margin: float = 0.3, lambda_triplet: float = 0.7):
        super().__init__()
        self.triplet = TripletLoss(margin)
        self.ce = nn.CrossEntropyLoss()
        self.lambda_triplet = lambda_triplet

    def forward(self, anchor_emb, pos_emb, neg_emb, logits, labels):
        triplet_loss, d_pos_mean, d_neg_mean = self.triplet(anchor_emb, pos_emb, neg_emb)
        ce_loss = self.ce(logits, labels)
        total = self.lambda_triplet * triplet_loss + (1 - self.lambda_triplet) * ce_loss
        return {
            'total_loss': total,
            'triplet_loss': triplet_loss.detach(),
            'ce_loss': ce_loss.detach(),
            'd_pos': d_pos_mean,
            'd_neg': d_neg_mean,
        }

__all__ = ["CombinedLoss"]
