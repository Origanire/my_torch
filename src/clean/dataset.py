"""Lecture d'un fichier d'échiquiers (une position FEN par ligne, label optionnel).

Parallélisme : les lignes sont découpées en morceaux convertis par plusieurs
processus (multiprocessing). Des processus et non des threads, car le GIL de
Python empêche deux threads d'exécuter du code Python en même temps.
"""

import os
from multiprocessing import Pool

import numpy as np

from .fen import INPUT_SIZE, fen_to_vector
from .labels import label_to_class, split_line

# En dessous, lancer des processus coûte plus cher que la conversion elle-même
PARALLEL_THRESHOLD = 20000


def encode_lines(args):
    """Convertit un morceau de lignes en (entrées uint8, classes ou None)."""
    lines, first_line_number, require_labels = args
    inputs = np.zeros((len(lines), INPUT_SIZE), dtype=np.uint8)
    targets = np.full(len(lines), -1, dtype=np.int64)
    for i, line in enumerate(lines):
        number = first_line_number + i
        try:
            fen, label = split_line(line)
            inputs[i] = fen_to_vector(fen, dtype=np.uint8)
            if require_labels:
                if label is None:
                    raise ValueError("missing expected output")
                targets[i] = label_to_class(label, fen)
        except ValueError as err:
            raise ValueError(f"line {number}: {err}") from None
    return inputs, targets


def load_chess_file(path, require_labels=False, workers=None):
    """Renvoie (inputs (N, 769) uint8, targets (N,) int64 ; -1 si pas de label)."""
    with open(path) as file:
        lines = [line for line in file.read().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"no chessboard in '{path}'")

    workers = workers or os.cpu_count() or 1
    if workers == 1 or len(lines) < PARALLEL_THRESHOLD:
        return encode_lines((lines, 1, require_labels))

    size = -(-len(lines) // workers)
    chunks = [(lines[i:i + size], i + 1, require_labels) for i in range(0, len(lines), size)]
    with Pool(workers) as pool:
        results = pool.map(encode_lines, chunks)
    return np.concatenate([r[0] for r in results]), np.concatenate([r[1] for r in results])


def load_cached(path, cache_dir=os.path.join("dataset", "cache")):
    """Comme load_chess_file, mais garde le résultat en .npy pour les benchmarks.

    Memory map : le fichier est projeté en mémoire et partagé entre processus au lieu d'être copié.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    x_path = os.path.join(cache_dir, f"{name}_inputs.npy")
    y_path = os.path.join(cache_dir, f"{name}_targets.npy")
    if not (os.path.exists(x_path) and os.path.getmtime(x_path) > os.path.getmtime(path)):
        os.makedirs(cache_dir, exist_ok=True)
        inputs, targets = load_chess_file(path, require_labels=True)
        np.save(y_path, targets)
        np.save(x_path, inputs)
    return np.load(x_path, mmap_mode="r"), np.load(y_path)
