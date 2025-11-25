import random
from typing import Optional
from PIL import Image, ImageEnhance
import numpy as np
import torchvision.transforms.functional as F

FACTOR_LIMIT = 40  # Límite superior de réplicas por imagen original

# --- Transformaciones básicas ---

def _rand_horizontal_flip(img: Image.Image, rng: random.Random) -> Image.Image:
    # Probabilidad aleatoria entre 0.4 y 0.6 de aplicar (ligera variación)
    if rng.random() < rng.uniform(0.4, 0.6):
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def _rand_brightness(img: Image.Image, rng: random.Random) -> Image.Image:
    factor = rng.uniform(0.85, 1.25)  # brillo moderado
    return ImageEnhance.Brightness(img).enhance(factor)


def _rand_contrast(img: Image.Image, rng: random.Random) -> Image.Image:
    factor = rng.uniform(0.85, 1.25)
    return ImageEnhance.Contrast(img).enhance(factor)


def _rand_saturation(img: Image.Image, rng: random.Random) -> Image.Image:
    factor = rng.uniform(0.85, 1.30)
    return ImageEnhance.Color(img).enhance(factor)


def _rand_rotation(img: Image.Image, rng: random.Random) -> Image.Image:
    # Ángulo en ±[5,10] grados excluyendo rango pequeño cercano a 0.
    angle = rng.uniform(5, 10)
    if rng.random() < 0.5:
        angle = -angle
    return img.rotate(angle, resample=Image.BILINEAR, expand=False)


def _rand_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.0
    mode = rng.choice(["gaussian", "poisson"])
    if mode == "gaussian":
        sigma = rng.uniform(0.005, 0.03)
        noise = rng.normalvariate(0, sigma)
        # Vectorizar ruido gaussian por canal
        gauss = np.random.normal(0, sigma, arr.shape)
        arr = arr + gauss
    else:  # poisson
        # Escalar para simular conteos y volver
        vals = rng.uniform(20, 60)
        arr = np.clip(arr, 0, 1)
        noisy = np.random.poisson(arr * vals) / float(vals)
        arr = noisy
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _rand_perspective(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    distortion_scale = rng.uniform(0.02, 0.10)
    # Puntos de entrada (esquinas) y salida (perturbadas)
    startpoints = [(0, 0), (w, 0), (w, h), (0, h)]
    def jitter(pt):
        x, y = pt
        dx = rng.uniform(-distortion_scale, distortion_scale) * w
        dy = rng.uniform(-distortion_scale, distortion_scale) * h
        return (x + dx, y + dy)
    endpoints = [jitter(pt) for pt in startpoints]
    return F.perspective(img, startpoints, endpoints, interpolation=Image.BICUBIC)

# Pool de constructores de transformaciones: cada devuelve imagen transformada.
_TRANSFORM_BUILDERS = [
    _rand_horizontal_flip,
    _rand_brightness,
    _rand_noise,
    _rand_rotation,
    _rand_contrast,
    _rand_saturation,
    _rand_perspective,
]


def augment_for_country(img: Image.Image, country_factor: float, rng: Optional[random.Random] = None) -> Image.Image:
    """Aplica subset aleatorio de 3–5 transformaciones si country_factor > 1.
    Parámetros de cada transformación se muestrean aleatoriamente.
    El orden de aplicación también es aleatorio.
    Si factor <= 1, retorna imagen original.
    """
    if country_factor <= 1:
        return img
    if rng is None:
        rng = random.Random()
    k = rng.randint(3, 5)
    chosen = rng.sample(_TRANSFORM_BUILDERS, k=k)
    # Mezclar orden
    rng.shuffle(chosen)
    out = img
    for fn in chosen:
        out = fn(out, rng)
    return out

__all__ = ["augment_for_country", "FACTOR_LIMIT"]
