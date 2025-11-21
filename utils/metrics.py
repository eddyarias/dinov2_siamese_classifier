import torch
import torch.nn.functional as F
from typing import Tuple

def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean().item()

def embedding_norm(emb: torch.Tensor) -> float:
    return emb.norm(p=2, dim=-1).mean().item()

def pairwise_knn_accuracy(embeddings: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    # embeddings: (N, D), labels: (N,)
    with torch.no_grad():
        # Distancia Euclidiana cuadrada
        dists = torch.cdist(embeddings, embeddings, p=2)  # (N,N)
        N = embeddings.size(0)
        correct = 0
        for i in range(N):
            # Excluir self
            row = dists[i]
            row[i] = float('inf')
            knn_idx = torch.topk(row, k, largest=False).indices
            knn_labels = labels[knn_idx]
            # voto mayoritario
            values, counts = torch.unique(knn_labels, return_counts=True)
            pred = values[torch.argmax(counts)]
            if pred == labels[i]:
                correct += 1
        return correct / N

__all__ = ["classification_accuracy", "embedding_norm", "pairwise_knn_accuracy"]
