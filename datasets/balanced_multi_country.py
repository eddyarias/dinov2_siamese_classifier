import os
import json
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from .data_agumentation import augment_for_country, FACTOR_LIMIT

class BalancedMultiCountryTripletDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        metadata_path: str,
        list_filename: str = "train.txt",
        transform=None,
        image_mode: str = "RGB",
        apply_balancing: bool = True,
    ):
        self.root_dir = root_dir
        self.metadata_path = metadata_path
        self.list_filename = list_filename
        self.transform = transform
        self.image_mode = image_mode
        self.apply_balancing = apply_balancing

        self.samples: List[Tuple[str, int, str]] = []  # (path, label_index, country)
        self.country_factors: Dict[str, float] = {}
        self.country_indices: Dict[str, List[int]] = {}
        self.class_indices: Dict[int, List[int]] = defaultdict(list)
        self.virtual_index: List[int] = []  # índice repetido según factor

        # Mapeo de nombres de clase a índices
        self.class_map = {
            'digital': 0,
            'border': 1,
            'printed': 2,
            'plastic': 2,   # plastic mapeado a misma clase que printed
            'screen': 3,
            'synthetic': 4,
        }

        self._load_metadata()
        self._load_country_lists()
        self._build_country_indices()
        self._build_virtual_index()

    def _load_metadata(self):
        if not os.path.isfile(self.metadata_path):
            raise FileNotFoundError(f"metadata.json no encontrado: {self.metadata_path}")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        cfg = meta.get('country_balancing_configuration')
        if cfg is None:
            raise ValueError("metadata.json: falta 'country_balancing_configuration'")
        ref = cfg.get('reference_country')
        if ref is None or 'subdirectory' not in ref:
            raise ValueError("metadata.json: falta 'reference_country.subdirectory'")
        reference_subdir = ref['subdirectory']
        self.country_factors[reference_subdir] = 1.0
        countries = cfg.get('countries', [])
        for c in countries:
            sub = c['subdirectory']
            factor = float(c.get('factor', 1.0))
            self.country_factors[sub] = factor

    def _parse_line(self, line: str) -> Tuple[str, int]:
        parts = line.strip().split()
        if len(parts) < 2:
            raise ValueError(f"Línea inválida (faltan tokens): '{line}'")
        label_token = parts[-1]
        path = " ".join(parts[:-1])  # conserva espacios en la ruta
        label_index: int | None = None
        # Intentar int
        if label_token.isdigit():
            label_index = int(label_token)
        else:
            lt = label_token.lower()
            if lt in self.class_map:
                label_index = self.class_map[lt]
        if label_index is None:
            raise ValueError(f"Etiqueta no reconocida '{label_token}' en línea: '{line}'")
        return path, label_index

    def _load_country_lists(self):
        # Recorrer subdirectorios definidos en factors con barra de progreso
        for country, _ in self.country_factors.items():
            list_path = os.path.join(self.root_dir, country, self.list_filename)
            if not os.path.isfile(list_path):
                raise FileNotFoundError(f"Lista para país '{country}' no encontrada: {list_path}")
            with open(list_path, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith('#')]
            for line in tqdm(lines, desc=f"Cargando {country}", leave=False):
                try:
                    img_path, label_idx = self._parse_line(line)
                except Exception:
                    # Ignorar línea malformada
                    continue
                if not os.path.isfile(img_path):
                    continue
                self.samples.append((img_path, label_idx, country))
        if len(self.samples) == 0:
            raise ValueError("No se cargaron muestras de ningún país (verifique formato de listas)")

    def _build_country_indices(self):
        for idx, (_, label, country) in enumerate(self.samples):
            self.country_indices.setdefault(country, []).append(idx)
            self.class_indices[label].append(idx)
        if len(self.country_indices) < 2:
            raise ValueError("Se requieren >=2 países para balanceo")

    def _build_virtual_index(self):
        if not self.apply_balancing:
            # Sin balanceo: usar cada muestra una sola vez
            self.virtual_index = list(range(len(self.samples)))
            return
        # Con balanceo: repetir índices según factor (entero aproximado)
        for country, indices in self.country_indices.items():
            factor = self.country_factors.get(country, 1.0)
            # Limitar número de réplicas según especificación (incluye la instancia original)
            repeat_times = min(max(int(round(factor)), 1), FACTOR_LIMIT)
            for _ in range(repeat_times):
                self.virtual_index.extend(indices)
        random.shuffle(self.virtual_index)

    def __len__(self):
        return len(self.virtual_index)

    def _sample_positive(self, anchor_idx: int, label: int) -> int:
        candidates = self.class_indices.get(label, [])
        if len(candidates) <= 1:
            return anchor_idx
        pos = anchor_idx
        # Reintentos limitados para evitar bucles
        for _ in range(10):
            pos = random.choice(candidates)
            if pos != anchor_idx:
                break
        return pos

    def _sample_negative(self, label: int) -> int:
        # Elegir cualquier índice cuyo label sea distinto
        # Optimizado: seleccionar clase distinta primero
        other_classes = [cls for cls in self.class_indices.keys() if cls != label]
        if not other_classes:
            return random.choice(self.class_indices.get(label, [0]))
        neg_class = random.choice(other_classes)
        neg_candidates = self.class_indices[neg_class]
        return random.choice(neg_candidates)

    def _load_image(self, path: str):
        try:
            img = Image.open(path)
            if self.image_mode:
                img = img.convert(self.image_mode)
            return img
        except Exception:
            return None

    def __getitem__(self, index: int) -> Dict[str, Any]:
        real_idx = self.virtual_index[index]
        anchor_path, anchor_label, anchor_country = self.samples[real_idx]
        pos_idx = self._sample_positive(real_idx, anchor_label)
        neg_idx = self._sample_negative(anchor_label)
        pos_path, _, pos_country = self.samples[pos_idx]
        neg_path, _, neg_country = self.samples[neg_idx]

        anchor_img = self._load_image(anchor_path)
        pos_img = self._load_image(pos_path)
        neg_img = self._load_image(neg_path)

        # Reemplazo simple si alguna imagen falla
        if anchor_img is None:
            for _ in range(5):
                alt_idx = random.randint(0, len(self.samples) - 1)
                alt_path, alt_label, alt_country = self.samples[alt_idx]
                anchor_img = self._load_image(alt_path)
                if anchor_img is not None:
                    anchor_label = alt_label
                    anchor_country = alt_country
                    anchor_path = alt_path
                    break
        if pos_img is None:
            for _ in range(5):
                pos_idx = self._sample_positive(real_idx, anchor_label)
                pos_path, _, _ = self.samples[pos_idx]
                pos_img = self._load_image(pos_path)
                if pos_img is not None:
                    break
        if neg_img is None:
            for _ in range(5):
                neg_idx = self._sample_negative(anchor_label)
                neg_path, _, _ = self.samples[neg_idx]
                neg_img = self._load_image(neg_path)
                if neg_img is not None:
                    break

        if anchor_img is None or pos_img is None or neg_img is None:
            raise RuntimeError("Fallo al cargar imágenes tras reintentos.")

        # Augmentación condicional por factor de país (solo si >1)
        if self.apply_balancing:
            anchor_factor = self.country_factors.get(anchor_country, 1.0)
            pos_factor = self.country_factors.get(pos_country, 1.0)
            neg_factor = self.country_factors.get(neg_country, 1.0)
            anchor_img = augment_for_country(anchor_img, anchor_factor)
            pos_img = augment_for_country(pos_img, pos_factor)
            neg_img = augment_for_country(neg_img, neg_factor)

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return {
            'anchor': anchor_img,
            'positive': pos_img,
            'negative': neg_img,
            'label': anchor_label,
            'country': anchor_country,
            'anchor_path': anchor_path,
        }

__all__ = ['BalancedMultiCountryTripletDataset']
