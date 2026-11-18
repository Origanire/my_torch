"""Vérifie l'encodage FEN, les labels et l'augmentation de données."""

import unittest

from src.clean.augment import augment_line, mirror, swap_colors, swap_label
from src.clean.fen import INPUT_SIZE, fen_to_vector
from src.clean.labels import CLASS_INDEX, label_to_class, split_line

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class FenTests(unittest.TestCase):
    def test_start_position_has_32_pieces_and_white_to_move(self):
        vector = fen_to_vector(START)
        self.assertEqual(vector.shape, (INPUT_SIZE,))
        self.assertEqual(int(vector[:-1].sum()), 32)
        self.assertEqual(vector[-1], 1.0)

    def test_invalid_fen_is_rejected(self):
        for fen in ("8/8/8 w", "9/8/8/8/8/8/8/8 w", "8/8/8/8/8/8/8/7X w", "8/8/8/8/8/8/8/8 x"):
            with self.subTest(fen=fen), self.assertRaises(ValueError):
                fen_to_vector(fen)


class LabelTests(unittest.TestCase):
    def test_split_line_separates_fen_and_label(self):
        self.assertEqual(split_line(START + " Checkmate White"), (START, "Checkmate White"))
        self.assertEqual(split_line(START), (START, None))

    def test_colourless_check_is_given_to_the_attacker(self):
        black_to_move = START.replace(" w ", " b ")
        self.assertEqual(label_to_class("Check", black_to_move), CLASS_INDEX["Check White"])
        self.assertEqual(label_to_class("Check", START), CLASS_INDEX["Check Black"])


class AugmentTests(unittest.TestCase):
    def test_transformations_are_involutions(self):
        fen = "4k3/8/8/8/8/8/3Q4/4K3 b - - 0 1"
        self.assertEqual(mirror(mirror(fen)), fen)
        self.assertEqual(swap_colors(swap_colors(fen)), fen)

    def test_swap_colors_swaps_the_label(self):
        self.assertEqual(swap_label("Checkmate White"), "Checkmate Black")
        self.assertEqual(swap_label("Nothing"), "Nothing")

    def test_augment_line_gives_four_variants(self):
        lines = augment_line("4k3/8/8/8/8/8/3Q4/4K3 b - - 0 1 Check White")
        self.assertEqual(len(lines), 4)
        self.assertEqual(sum(line.endswith("Check White") for line in lines), 2)
        self.assertEqual(sum(line.endswith("Check Black") for line in lines), 2)


if __name__ == "__main__":
    unittest.main()
