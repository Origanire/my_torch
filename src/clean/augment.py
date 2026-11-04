"""Augmentation de données par symétries de l'échiquier.

Augmentation : créer de nouveaux exemples valides à partir des existants, en
appliquant des transformations qui ne changent pas la bonne réponse (ou la
changent de façon connue). Plus d'exemples variés = moins de surapprentissage.
  - miroir gauche-droite (a <-> h) : échec et mat sont conservés, label inchangé ;
  - échange des couleurs (plateau retourné, pièces et trait inversés) : White <-> Black.
Roque et prise en passant sont mis à "-" : ils ne seraient plus cohérents.

Usage : python3 -m src.clean.augment ENTREE SORTIE
"""

import sys

from .labels import split_line

SWAP_COLOR = {"White": "Black", "Black": "White"}


def expand(row):
    """'2r2b1k' -> '..r..b.k' (une lettre ou un point par case)."""
    return "".join("." * int(c) if c.isdigit() else c for c in row)


def compress(row):
    """'..r..b.k' -> '2r2b1k'."""
    out, empty = "", 0
    for c in row:
        if c == ".":
            empty += 1
        else:
            out += (str(empty) if empty else "") + c
            empty = 0
    return out + (str(empty) if empty else "")


def mirror(fen):
    """Symétrie gauche-droite du plateau."""
    fields = fen.split()
    rows = [compress(expand(row)[::-1]) for row in fields[0].split("/")]
    return " ".join(["/".join(rows), fields[1], "-", "-"] + fields[4:])


def swap_colors(fen):
    """Retourne le plateau et échange les couleurs des pièces et du trait."""
    fields = fen.split()
    rows = [row.swapcase() for row in reversed(fields[0].split("/"))]
    side = "b" if fields[1] == "w" else "w"
    return " ".join(["/".join(rows), side, "-", "-"] + fields[4:])


def swap_label(label):
    """'Check White' -> 'Check Black' ; 'Nothing' reste 'Nothing'."""
    return " ".join(SWAP_COLOR.get(word, word) for word in label.split())


def augment_line(line):
    """Renvoie les 4 variantes d'une ligne : originale, miroir, couleurs échangées, les deux."""
    fen, label = split_line(line)
    swapped_fen, swapped_label = swap_colors(fen), swap_label(label)
    return [f"{fen} {label}", f"{mirror(fen)} {label}",
            f"{swapped_fen} {swapped_label}", f"{mirror(swapped_fen)} {swapped_label}"]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        sys.exit(84)
    with open(sys.argv[1]) as src, open(sys.argv[2], "w") as dst:
        for line in src:
            if line.strip():
                dst.write("\n".join(augment_line(line)) + "\n")
