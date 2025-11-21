from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

class TensorBoardLogger:
    def __init__(self, base_dir: str):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(base_dir, f'run_{ts}')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def log_step(self, step: int, metrics: dict):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, step)

    def log_histogram(self, step: int, name: str, values):
        self.writer.add_histogram(name, values, step)

    def close(self):
        self.writer.flush()
        self.writer.close()

__all__ = ["TensorBoardLogger"]
