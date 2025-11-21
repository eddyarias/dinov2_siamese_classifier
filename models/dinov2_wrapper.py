import torch
import torch.nn as nn
import timm
from typing import Optional, Dict, Any

class EmbeddingHead(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int):
        super().__init__()
        hidden = max(emb_dim * 2, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        # L2 normalize embeddings
        return nn.functional.normalize(z, p=2, dim=-1)

class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        hidden = max(in_dim // 2, 256)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DINOv2SiameseModel(nn.Module):
    """Wrapper para backbone DINOv2 + heads embedding y clasificación.
    Forward puede recibir anchor, positive, negative.
    Clasificación solo se calcula sobre anchor (etiqueta principal).
    """
    def __init__(
        self,
        backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
        embedding_dim: int = 384,
        num_classes: int = 5,
        unfreeze_blocks: int = 2,
        img_size: int = 224,
    ):
        super().__init__()
        # Crear backbone con tamaño de imagen controlado.
        # Para modelos ViT de timm esto ajusta la expectativa de HxW de entrada.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            img_size=img_size,
        )  # sin clasificación interna
        feat_dim = self.backbone.num_features
        self.embedding_head = EmbeddingHead(feat_dim, embedding_dim)
        self.classification_head = ClassificationHead(feat_dim, num_classes)
        self._freeze_except_last(unfreeze_blocks)

    def _freeze_except_last(self, n_blocks: int):
        # Congelar todo inicialmente
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Descongelar últimos n bloques si existe atributo blocks
        if hasattr(self.backbone, 'blocks') and isinstance(self.backbone.blocks, nn.ModuleList):
            total = len(self.backbone.blocks)
            to_unfreeze = self.backbone.blocks[total - n_blocks:]
            for blk in to_unfreeze:
                for p in blk.parameters():
                    p.requires_grad = True
        else:
            # Si no hay blocks, descongelar todo (fallback)
            for p in self.backbone.parameters():
                p.requires_grad = True

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: Optional[torch.Tensor] = None,
        negative: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        anchor_feat = self.encode(anchor)
        anchor_emb = self.embedding_head(anchor_feat)
        anchor_logits = self.classification_head(anchor_feat)

        out: Dict[str, Any] = {
            'anchor_embedding': anchor_emb,
            'anchor_logits': anchor_logits,
        }

        if positive is not None:
            pos_feat = self.encode(positive)
            pos_emb = self.embedding_head(pos_feat)
            out['positive_embedding'] = pos_emb
        if negative is not None:
            neg_feat = self.encode(negative)
            neg_emb = self.embedding_head(neg_feat)
            out['negative_embedding'] = neg_emb
        return out

__all__ = ['DINOv2SiameseModel', 'EmbeddingHead', 'ClassificationHead']
