"""Transforme une position FEN en vecteur de 769 nombres pour le réseau."""

import numpy as np

# One-hot : un "plan" 8x8 par type de pièce, 1 si la pièce est sur la case, 0 sinon
PIECES = "PNBRQKpnbrqk"
PIECE_INDEX = {piece: i for i, piece in enumerate(PIECES)}

INPUT_SIZE = 12 * 64 + 1


def fen_to_vector(fen):
    """Renvoie le vecteur : 12 plans x 64 cases, puis le trait (1 = blancs, 0 = noirs)."""
    fields = fen.split()
    if len(fields) < 2:
        raise ValueError(f"invalid FEN: '{fen}'")
    board, side = fields[0], fields[1]

    vector = np.zeros(INPUT_SIZE, dtype=np.float32)
    rows = board.split("/")
    if len(rows) != 8:
        raise ValueError(f"invalid FEN board: '{board}'")

    for rank, row in enumerate(rows):
        file = 0
        for char in row:
            if char.isdigit():
                # Un chiffre = autant de cases vides
                file += int(char)
            elif char in PIECE_INDEX:
                if file >= 8:
                    raise ValueError(f"invalid FEN row: '{row}'")
                vector[PIECE_INDEX[char] * 64 + rank * 8 + file] = 1.0
                file += 1
            else:
                raise ValueError(f"invalid FEN character: '{char}'")
        if file != 8:
            raise ValueError(f"invalid FEN row: '{row}'")

    if side not in ("w", "b"):
        raise ValueError(f"invalid side to move: '{side}'")
    vector[-1] = 1.0 if side == "w" else 0.0
    return vector
