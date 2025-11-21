import os
import random
from typing import List, Tuple, Dict, Any
from PIL import Image

import torch
from torch.utils.data import Dataset


def triplet_collate(batch: List[Dict[str, Any]]):
    anchors = torch.stack([b["anchor"] for b in batch])
    positives = torch.stack([b["positive"] for b in batch])
    negatives = torch.stack([b["negative"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    countries = [b.get("country", None) for b in batch]
    return {
        "anchors": anchors,
        "positives": positives,
        "negatives": negatives,
        "labels": labels,
        "countries": countries,
    }

__all__ = ["ImageTripletDataset", "triplet_collate"]
