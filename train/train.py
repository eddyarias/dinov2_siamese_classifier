import os
import yaml
import time
import json
from typing import Dict, Any

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from datasets.image_list_dataset import triplet_collate
from datasets.balanced_multi_country import BalancedMultiCountryTripletDataset
from models.dinov2_wrapper import DINOv2SiameseModel
from losses.combined import CombinedLoss
from utils.metrics import classification_accuracy, embedding_norm
from utils.logging import TensorBoardLogger


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def select_device(cfg: Dict[str, Any]):
    if cfg.get('device', 'auto') == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg['device']


def build_transforms(cfg: Dict[str, Any]):
    # Tamaño controlado desde config (data.size_img) con fallback 224
    size = cfg.get('data', {}).get('size_img', 224)
    train_tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return train_tf, val_tf


def create_dataloaders(cfg: Dict[str, Any]):
    """
    Construye dataloaders siempre en modo multi-país usando countries_root + metadata.
    Requiere que cada subdirectorio contenga 'train.txt' y 'validation.txt'.
    """
    train_tf, val_tf = build_transforms(cfg)
    root_dir = cfg['paths']['countries_root']
    metadata_path = cfg['paths']['metadata']
    if not (os.path.isdir(root_dir) and os.path.isfile(metadata_path)):
        raise FileNotFoundError("Modo multi-país activo: se requiere 'countries_root' directorio y archivo 'metadata'.")
    train_ds = BalancedMultiCountryTripletDataset(root_dir=root_dir, metadata_path=metadata_path, list_filename='train.txt', transform=train_tf)
    val_ds = BalancedMultiCountryTripletDataset(root_dir=root_dir, metadata_path=metadata_path, list_filename='validation.txt', transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,  # mezcla global del índice virtual balanceado
        num_workers=cfg['training']['num_workers'],
        pin_memory=cfg['training']['pin_memory'],
        persistent_workers=cfg['training']['persistent_workers'],
        collate_fn=triplet_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=cfg['training']['pin_memory'],
        persistent_workers=cfg['training']['persistent_workers'],
        collate_fn=triplet_collate,
    )
    return train_loader, val_loader


def build_model(cfg: Dict[str, Any]):
    mcfg = cfg['model']
    img_size = cfg.get('data', {}).get('size_img', 224)
    model = DINOv2SiameseModel(
        backbone_name=mcfg['backbone_name'],
        embedding_dim=mcfg['embedding_dim'],
        num_classes=mcfg['num_classes'],
        unfreeze_blocks=mcfg['unfreeze_blocks'],
        img_size=img_size,
    )
    return model


def build_optimizer_scheduler(model, cfg):
    ocfg = cfg['optimizer']
    params = [p for p in model.parameters() if p.requires_grad]
    # Normalizar tipos (pueden venir como strings del YAML)
    lr = float(ocfg['lr'])
    weight_decay = float(ocfg.get('weight_decay', 0.0))
    betas_raw = ocfg.get('betas', [0.9, 0.999])
    betas = tuple(float(b) for b in betas_raw)
    if ocfg['name'].lower() == 'adamw':
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    else:
        optimizer = optim.Adam(params, lr=lr, betas=betas)

    scfg = cfg['scheduler']
    if scfg['name'] == 'cosine':
        t_max = int(scfg['t_max'])
        eta_min = float(scfg['eta_min'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    else:
        scheduler = None
    return optimizer, scheduler


def save_checkpoint(state: Dict[str, Any], path: str):
    torch.save(state, path)


def validate(model, val_loader, device):
    model.eval()
    total_acc = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            anchors = batch['anchors'].to(device)
            labels = batch['labels'].to(device)
            out = model(anchor=anchors)
            logits = out['anchor_logits']
            acc = classification_accuracy(logits, labels)
            total_acc += acc
            total_batches += 1
    return total_acc / max(total_batches, 1)


def train(cfg_path: str = 'configs/config.yaml'):
    cfg = load_config(cfg_path)
    device = select_device(cfg)
    # Directorios base
    os.makedirs(cfg['paths']['checkpoints_dir'], exist_ok=True)
    os.makedirs(cfg['paths']['logs_dir'], exist_ok=True)

    # Timestamp para esta corrida
    run_ts = time.strftime('%Y%m%d_%H%M%S')
    run_checkpoint_dir = os.path.join(cfg['paths']['checkpoints_dir'], f'checkpoint_{run_ts}')
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    train_loader, val_loader = create_dataloaders(cfg)

    # Crear snapshot de parámetros y resumen dataset
    try:
        train_ds = train_loader.dataset  # type: ignore
        ds_summary = {
            'num_train_samples': len(train_ds),
            'train_country_counts': {c: len(ixs) for c, ixs in train_ds.country_indices.items()},
            'country_factors': train_ds.country_factors,
            'num_virtual_index': len(train_ds.virtual_index)
        }
    except Exception:
        ds_summary = {
            'num_train_samples': None,
            'train_country_counts': None,
            'country_factors': None,
            'num_virtual_index': None
        }

    snapshot = {
        'timestamp': run_ts,
        'config_path': cfg_path,
        'device': device,
        'torch_version': torch.__version__,
        'config': cfg,
        'dataset_summary': ds_summary
    }
    snapshot_name = f'snapshot_parameters_{run_ts}.json'
    snapshot_path = os.path.join(run_checkpoint_dir, snapshot_name)
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    model = build_model(cfg).to(device)
    # Activar gradient checkpointing opcional
    if cfg['training'].get('enable_checkpointing') and hasattr(model.backbone, 'set_grad_checkpointing'):
        model.backbone.set_grad_checkpointing(True)
    loss_fn = CombinedLoss(
        margin=cfg['losses']['triplet_margin'],
        lambda_triplet=cfg['losses']['lambda_triplet']
    )
    optimizer, scheduler = build_optimizer_scheduler(model, cfg)

    logger = TensorBoardLogger(cfg['paths']['logs_dir'])

    # AMP API actualizada (FutureWarning): usar torch.amp.GradScaler y torch.amp.autocast
    device_type = 'cuda' if device.startswith('cuda') else 'cpu'
    # GradScaler nueva API: primer argumento posicional = device ('cuda'/'cpu'). En versiones actuales solo 'cuda' habilita escalado.
    scaler = torch.amp.GradScaler(device_type, enabled=cfg['training']['mixed_precision'] and device_type == 'cuda')
    epochs = cfg['training']['epochs']
    global_step = 0

    best_val = -float('inf')
    best_ckpt_path = os.path.join(run_checkpoint_dir, 'best.pth')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        accum_steps = int(cfg['training'].get('gradient_accumulation_steps', 1))
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for batch_idx, batch in enumerate(train_iter, start=1):
            anchors = batch['anchors'].to(device)
            positives = batch['positives'].to(device)
            negatives = batch['negatives'].to(device)
            labels = batch['labels'].to(device)

            # Reiniciar gradientes al inicio de cada grupo de acumulación
            if (batch_idx - 1) % accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_type, enabled=cfg['training']['mixed_precision'] and device_type == 'cuda'):
                out = model(anchor=anchors, positive=positives, negative=negatives)
                losses = loss_fn(out['anchor_embedding'], out['positive_embedding'], out['negative_embedding'], out['anchor_logits'], labels)
                total_loss = losses['total_loss']
                scaled_loss = total_loss / accum_steps

            scaler.scale(scaled_loss).backward()
            if batch_idx % accum_steps == 0:
                if cfg['training']['grad_clip'] is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            acc = classification_accuracy(out['anchor_logits'], labels)
            emb_norm = embedding_norm(out['anchor_embedding'])

            step_metrics = {
                'train/total_loss': total_loss.item(),
                'train/triplet_loss': losses['triplet_loss'].item(),
                'train/ce_loss': losses['ce_loss'].item(),
                'train/acc': acc,
                'train/pos_dist': losses['d_pos'],
                'train/neg_dist': losses['d_neg'],
                'train/emb_norm': emb_norm,
                'lr': optimizer.param_groups[0]['lr'],
            }
            logger.log_step(global_step, step_metrics)
            train_iter.set_postfix({'loss': f"{total_loss.item():.3f}", 'acc': f"{acc:.3f}"})
            global_step += 1
            epoch_loss_sum += total_loss.item()
            epoch_loss_count += 1

        # Validación con barra
        model.eval()
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False)
            for batch in val_iter:
                anchors = batch['anchors'].to(device)
                labels = batch['labels'].to(device)
                out = model(anchor=anchors)
                logits = out['anchor_logits']
                acc = classification_accuracy(logits, labels)
                val_total += acc
                val_batches += 1
                val_iter.set_postfix({'acc': f"{acc:.3f}"})
        val_acc = val_total / max(val_batches, 1)
        logger.log_step(global_step, {'val/accuracy': val_acc})

        if val_acc > best_val:
            best_val = val_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict() if scheduler else None,
                'val_accuracy': val_acc,
            }, best_ckpt_path)

        # Guardar checkpoint de la época
        epoch_ckpt = os.path.join(run_checkpoint_dir, f'epoch_{epoch}.pth')
        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'val_accuracy': val_acc,
        }, epoch_ckpt)

        avg_train_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        print(f"Epoch {epoch}/{epochs} - val_acc={val_acc:.4f} - best={best_val:.4f} - train_loss={avg_train_loss:.4f} - time={(time.time()-epoch_start):.1f}s")

    logger.close()

if __name__ == '__main__':
    train()
