"""Fusionne les fichiers de dataset/ et les sépare en un jeu d'entraînement et un jeu de test.

Le jeu de test n'est jamais vu pendant l'entraînement : il mesure honnêtement
la capacité du réseau à généraliser sur des positions inconnues.

Usage : python3 -m src.clean.build_dataset [dossier_source] [proportion_test] [graine]
"""

import glob
import os
import random
import sys


def build(source="dataset", test_ratio=0.1, seed=42):
    """Écrit <source>/train.txt et <source>/test.txt (mélangés, graine fixe = reproductible)."""
    lines = []
    for path in sorted(glob.glob(os.path.join(source, "*", "*.txt"))):
        with open(path) as file:
            lines += [line for line in file.read().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"no data found in '{source}/*/*.txt'")
    random.Random(seed).shuffle(lines)
    n_test = int(len(lines) * test_ratio)
    for name, part in (("test.txt", lines[:n_test]), ("train.txt", lines[n_test:])):
        with open(os.path.join(source, name), "w") as file:
            file.write("\n".join(part) + "\n")
    print(f"{len(lines)} positions -> train: {len(lines) - n_test}, test: {n_test}")


if __name__ == "__main__":
    args = sys.argv[1:]
    build(args[0] if args else "dataset",
          float(args[1]) if len(args) > 1 else 0.1,
          int(args[2]) if len(args) > 2 else 42)
