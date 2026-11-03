"""Convert a FEN chess position into a vector of numbers for the network.

Encoding (769 values):
  - 12 planes of 8x8 (one per piece type, one-hot): 768 values
  - 1 value for the side to move (1 = white, 0 = black)
"""

import numpy as np

# Order of the 12 planes: white pieces then black pieces
PIECES = "PNBRQKpnbrqk"
PIECE_INDEX = {piece: i for i, piece in enumerate(PIECES)}

INPUT_SIZE = 12 * 64 + 1


def fen_to_vector(fen):
    """Return a numpy array of shape (769,) describing the position."""
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
                file += int(char)  # a digit = that many empty squares
            elif char in PIECE_INDEX:
                if file >= 8:
                    raise ValueError(f"invalid FEN row: '{row}'")
                square = rank * 8 + file
                vector[PIECE_INDEX[char] * 64 + square] = 1.0
                file += 1
            else:
                raise ValueError(f"invalid FEN character: '{char}'")
        if file != 8:
            raise ValueError(f"invalid FEN row: '{row}'")

    if side not in ("w", "b"):
        raise ValueError(f"invalid side to move: '{side}'")
    vector[-1] = 1.0 if side == "w" else 0.0
    return vector
