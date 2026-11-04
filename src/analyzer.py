"""my_torch_analyzer : entraîne un réseau ou prédit l'état de parties d'échecs."""

import sys

from src.clean.dataset import load_chess_file
from src.clean.fen import INPUT_SIZE
from src.clean.labels import CLASSES
from src.mlp.network import Network
from src.mlp.trainer import train

USAGE = """USAGE
    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE

DESCRIPTION
    --train     Launch the neural network in training mode. Each chessboard in FILE must
                contain inputs to send to the neural network in FEN notation and the expected output
                separated by space. If specified, the newly trained neural network will be saved in
                SAVEFILE. Otherwise, it will be saved in the original LOADFILE.

    --predict   Launch the neural network in prediction mode. Each chessboard in FILE must
                contain inputs to send to the neural network in FEN notation, and optionally an
                expected output.

    --save      Save neural network into SAVEFILE. Only works in train mode.

    LOADFILE    File containing an artificial neural network
    CHESSFILE   File containing chessboards"""


def parse_args(argv):
    """Renvoie (mode, loadfile, chessfile, savefile) ; lève ValueError si les arguments sont faux."""
    args = list(argv)
    mode = savefile = None
    positional = []
    while args:
        arg = args.pop(0)
        if arg in ("--predict", "--train"):
            if mode:
                raise ValueError("only one of --predict and --train can be given")
            mode = arg[2:]
        elif arg == "--save":
            if not args or savefile:
                raise ValueError("--save needs exactly one SAVEFILE")
            savefile = args.pop(0)
        else:
            positional.append(arg)
    if mode is None:
        raise ValueError("--predict or --train is required")
    if savefile and mode != "train":
        raise ValueError("--save only works in train mode")
    if len(positional) != 2:
        raise ValueError("LOADFILE and CHESSFILE are required")
    return mode, positional[0], positional[1], savefile


def check_shape(network):
    """Le réseau doit prendre un échiquier encodé et sortir une valeur par classe."""
    net = network.config["network"]
    if net["input_size"] != INPUT_SIZE or net["output_size"] != len(CLASSES):
        raise ValueError(f"network must have input_size={INPUT_SIZE} and output_size={len(CLASSES)}")


def run(argv):
    mode, loadfile, chessfile, savefile = parse_args(argv)
    network = Network.load(loadfile)
    check_shape(network)
    inputs, targets = load_chess_file(chessfile, require_labels=(mode == "train"))
    if mode == "predict":
        print("\n".join(CLASSES[i] for i in network.predict(inputs)))
    else:
        train(network, inputs, targets, network.config["training"])
        network.save(savefile or loadfile)


def main(argv):
    if len(argv) == 1 and argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    try:
        run(argv)
    except (ValueError, OSError) as err:
        print(f"my_torch_analyzer: {err}", file=sys.stderr)
        return 84
    return 0
