"""Correspondance entre les labels texte et les numéros de classe du réseau."""

CLASSES = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}

LABEL_WORDS = {"Nothing", "Check", "Checkmate", "Stalemate", "White", "Black"}


def split_line(line):
    """Sépare une ligne en (fen, label) ; label vaut None s'il est absent."""
    tokens = line.split()
    end = len(tokens)
    # Les mots de label sont à la fin de la ligne et ne peuvent pas apparaître dans un FEN
    while end > 0 and tokens[end - 1] in LABEL_WORDS:
        end -= 1
    # Un FEN a au plus 6 champs : ce qui suit est le label, même s'il est inconnu
    end = min(end, 6)
    fen = " ".join(tokens[:end])
    label = " ".join(tokens[end:]) or None
    return fen, label


def label_to_class(label, fen):
    """Numéro de classe d'un label. Sans couleur, on la déduit du trait :
    le camp qui joue subit l'échec, donc l'attaquant est l'autre camp."""
    if label in CLASS_INDEX:
        return CLASS_INDEX[label]
    if label in ("Check", "Checkmate"):
        fields = fen.split()
        attacker = "White" if len(fields) > 1 and fields[1] == "b" else "Black"
        return CLASS_INDEX[f"{label} {attacker}"]
    raise ValueError(f"unknown label: '{label}'")
