import os
import sys
import json
import argparse
import yaml
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Ensure project root (parent directory) is on sys.path when running directly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.balanced_multi_country import BalancedMultiCountryTripletDataset
from datasets.image_list_dataset import triplet_collate
from models.dinov2_wrapper import DINOv2SiameseModel


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_transforms(cfg: Dict[str, Any]):
    size = cfg.get('data', {}).get('size_img', 224)
    # Sin aumentos para evaluación
    eval_tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return eval_tf


def load_model_from_checkpoint(ckpt_path: str, cfg: Dict[str, Any], device: str):
    mcfg = cfg['model']
    img_size = cfg.get('data', {}).get('size_img', 224)
    model = DINOv2SiameseModel(
        backbone_name=mcfg['backbone_name'],
        embedding_dim=mcfg['embedding_dim'],
        num_classes=mcfg['num_classes'],
        unfreeze_blocks=mcfg['unfreeze_blocks'],
        img_size=img_size,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state'])
    model.eval()
    return model


def compute_embeddings(
    model,
    loader,
    device: str,
    collect_embeddings: bool = True,
    max_eval_samples: int | None = None,
    amp: bool = False,
    report_every: int = 0,
):
    """Devuelve (embeds, logits, labels, preds, countries). Si collect_embeddings=False,
    'embeds' será None y se saltan gráficos dependientes de embeddings.
    max_eval_samples limita el número total de muestras procesadas (para fast mode)."""
    device_type = 'cuda' if device.startswith('cuda') else 'cpu'
    all_embeds = []
    all_logits = []
    all_labels = []
    all_countries: list[str] = []
    seen = 0
    model.eval()
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            anchors = batch['anchors'].to(device)
            labels = batch['labels'].to(device)
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp and device_type=='cuda'):
                out = model(anchor=anchors)
            logits_cpu = out['anchor_logits'].cpu()
            all_logits.append(logits_cpu)
            all_labels.append(labels.cpu())
            all_countries.extend(batch['countries'])
            if collect_embeddings:
                all_embeds.append(out['anchor_embedding'].cpu())
            seen += anchors.size(0)
            if report_every > 0 and (b_idx + 1) % report_every == 0:
                print(f"[Eval] Procesadas {seen} muestras...")
            if max_eval_samples is not None and seen >= max_eval_samples:
                break
    logits = torch.cat(all_logits, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()
    preds = logits.argmax(axis=1)
    embeds_np = torch.cat(all_embeds, dim=0).numpy() if collect_embeddings and all_embeds else None
    return embeds_np, logits, labels_np, preds, np.array(all_countries)


def plot_confusion(labels, preds, class_names: List[str], out_path: str, title: str):
    cm = confusion_matrix(labels, preds)
    row_sums = cm.sum(axis=1, keepdims=True)
    # Evitar división por cero (filas sin ejemplos)
    row_sums[row_sums == 0] = 1
    cm_pct = (cm / row_sums) * 100.0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    ax.set_title(f'{title} (Percent)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            ax.text(j, i, f'{cm_pct[i, j]:.1f}%', ha='center', va='center', color='black')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('% within true class')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return cm_pct


def plot_embeddings_2d(embeds: np.ndarray, labels: np.ndarray, class_names: List[str], out_path: str, method: str = 'pca'):
    if method == 'tsne':
        reducer = TSNE(n_components=2, init='pca', random_state=42, perplexity=min(30, max(5, embeds.shape[0] // 50)))
    else:
        reducer = PCA(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeds)
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls_idx, cls_name in enumerate(class_names):
        cls_mask = labels == cls_idx
        ax.scatter(coords[cls_mask, 0], coords[cls_mask, 1], s=10, alpha=0.6, label=cls_name)
    ax.set_title(f'{method.upper()} Embeddings')
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_classification_report(labels, preds, class_names: List[str], out_txt: str):
    """Guarda solo el reporte en texto (sin versión JSON). Fuerza todas las clases esperadas."""
    num_classes = len(class_names)
    report_str = classification_report(labels, preds, target_names=class_names, labels=list(range(num_classes)), zero_division=0)
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(report_str)
    return report_str


def compute_distance_histograms(embeds: np.ndarray, labels: np.ndarray, out_dir: str):
    # Submuestreo para evitar O(n^2) grande
    max_samples = 1500
    if embeds.shape[0] > max_samples:
        idx = np.random.RandomState(42).choice(embeds.shape[0], max_samples, replace=False)
        embeds_sub = embeds[idx]
        labels_sub = labels[idx]
    else:
        embeds_sub = embeds
        labels_sub = labels
    # Distancias intra e inter
    intra = []
    inter = []
    for i in range(embeds_sub.shape[0]):
        for j in range(i + 1, embeds_sub.shape[0]):
            d = np.linalg.norm(embeds_sub[i] - embeds_sub[j])
            if labels_sub[i] == labels_sub[j]:
                intra.append(d)
            else:
                inter.append(d)
    intra = np.array(intra)
    inter = np.array(inter)
    # Graficar
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(intra, bins=40, alpha=0.7, label='Intra-class')
    ax.hist(inter, bins=40, alpha=0.7, label='Inter-class')
    ax.set_title('Distance Distributions (Intra vs Inter)')
    ax.set_xlabel('L2 Distance')
    ax.set_ylabel('Count')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hist_distance_intra_inter.png'))
    plt.close(fig)
    # Sin guardado de arrays .npy (solicitud: no npy)


def main():
    parser = argparse.ArgumentParser(description='Evaluación y métricas sobre conjunto de prueba (rápida opcional).')
    parser.add_argument('--checkpoint_dir', type=Path, required=True, help='Directorio de la corrida (checkpoint_YYYYMMDD_HHMMSS)')
    parser.add_argument('--list_name', type=str, default='test.txt', help='Nombre del archivo de lista para evaluación (default test.txt)')
    parser.add_argument('--method_2d', type=str, choices=['pca', 'tsne'], default='pca', help='Método para visualización 2D')
    parser.add_argument('--no_tsne_fallback', action='store_true', help='No intentar TSNE si se elige pca (solo pca).')
    parser.add_argument('--fast', action='store_true', help='Modo rápido: limita muestras, desactiva TSNE extra y métricas por país.')
    parser.add_argument('--skip_embeddings', action='store_true', help='No calcular embeddings (solo logits y métricas de clasificación).')
    parser.add_argument('--max_eval_samples', type=int, default=None, help='Máximo de muestras a evaluar (fast).')
    parser.add_argument('--eval_batch_size', type=int, default=None, help='Batch size para evaluación (sobrescribe config).')
    parser.add_argument('--amp', action='store_true', help='Usar autocast fp16 para acelerar en GPU.')
    parser.add_argument('--report_every', type=int, default=0, help='Imprimir progreso cada N lotes.')
    args = parser.parse_args()

    # Construir nombre de snapshot: snapshot_parameters_YYYYMMDD_HHMMSS.json
    run_id = args.checkpoint_dir.name.split('checkpoint_')[-1]
    snapshot_filename = f'snapshot_parameters_{run_id}.json'
    snapshot_path = args.checkpoint_dir / snapshot_filename
    if not snapshot_path.is_file():
        raise FileNotFoundError(f'Snapshot no encontrado: {snapshot_path}')
    with open(snapshot_path, 'r', encoding='utf-8') as f:
        snap = json.load(f)
    cfg_path = snap['config_path']
    cfg = load_config(cfg_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Usando dispositivo: {device}')
    eval_tf = build_transforms(cfg)

    root_dir = cfg['paths']['countries_root']
    metadata_path = cfg['paths']['metadata']
    ds = BalancedMultiCountryTripletDataset(root_dir=root_dir, metadata_path=metadata_path, list_filename=args.list_name, transform=eval_tf, apply_balancing=False)

    eval_batch_size = args.eval_batch_size or cfg['training']['batch_size']
    loader = DataLoader(
        ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=cfg['training']['pin_memory'],
        persistent_workers=cfg['training']['persistent_workers'],
        collate_fn=triplet_collate,
    )

    best_ckpt = args.checkpoint_dir / 'best.pth'
    if not best_ckpt.is_file():
        raise FileNotFoundError(f'best.pth no encontrado en {args.checkpoint_dir}')
    model = load_model_from_checkpoint(str(best_ckpt), cfg, device)

    # Determinar límites fast
    max_eval_samples = args.max_eval_samples
    if args.fast and max_eval_samples is None:
        # heurística: procesar como máximo 5000 muestras en modo rápido si dataset es mayor
        max_eval_samples = 5000
    embeds, logits, labels, preds, countries = compute_embeddings(
        model,
        loader,
        device,
        collect_embeddings=not args.skip_embeddings,
        max_eval_samples=max_eval_samples,
        amp=args.amp,
        report_every=args.report_every,
    )

    # Preparar carpeta metrics
    metrics_dir = args.checkpoint_dir / 'metrics'
    metrics_dir.mkdir(exist_ok=True)

    # Nombres de clases (asumimos 0..num_classes-1)
    num_classes = cfg['model']['num_classes']
    class_names = [f'class_{i}' for i in range(num_classes)]

    # Global metrics
    cm = plot_confusion(labels, preds, class_names, str(metrics_dir / 'confusion_all.png'), 'Confusion Matrix (All Countries)')
    report = save_classification_report(labels, preds, class_names, str(metrics_dir / 'classification_report_all.txt'))

    # 2D embedding plot
    if embeds is not None:
        plot_embeddings_2d(embeds, labels, class_names, str(metrics_dir / f'{args.method_2d}_all.png'), method=args.method_2d)
        if (not args.skip_embeddings) and args.method_2d == 'pca' and not args.no_tsne_fallback and not args.fast:
            try:
                plot_embeddings_2d(embeds, labels, class_names, str(metrics_dir / 'tsne_all.png'), method='tsne')
            except Exception:
                pass
        if not args.fast:
            compute_distance_histograms(embeds, labels, str(metrics_dir))

    # Por país
    unique_countries = sorted(np.unique(countries))
    if not args.fast:  # saltar métricas por país en modo rápido
        for ct in unique_countries:
            ct_mask = countries == ct
            ct_dir = metrics_dir / f'country_{ct}'
            ct_dir.mkdir(exist_ok=True)
            ct_labels = labels[ct_mask]
            ct_preds = preds[ct_mask]
            if len(ct_labels) == 0:
                continue
            plot_confusion(ct_labels, ct_preds, class_names, str(ct_dir / 'confusion.png'), f'Confusion ({ct})')
            save_classification_report(ct_labels, ct_preds, class_names, str(ct_dir / 'classification_report.txt'))
            if embeds is not None:
                ct_embeds = embeds[ct_mask]
                plot_embeddings_2d(ct_embeds, ct_labels, class_names, str(ct_dir / f'{args.method_2d}.png'), method=args.method_2d)

    # Sin archivo summary.json (solicitud: no summary). Sólo imprimir resumen.
    print('Evaluación completada. Métricas en:', metrics_dir)
    print('Resumen: samples=', labels.shape[0], 'countries=', unique_countries, 'accuracy=', float((preds == labels).mean()), 'fast_mode=', args.fast)
    if max_eval_samples is not None:
        print(f'Limitación de muestras aplicada: {labels.shape[0]} / {max_eval_samples}')


if __name__ == '__main__':
    main()
