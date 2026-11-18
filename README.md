# my_torch

[![Tests](https://github.com/Origanire/my_torch/actions/workflows/tests.yml/badge.svg)](https://github.com/Origanire/my_torch/actions/workflows/tests.yml)

> **English summary.** Neural network written from scratch with NumPy only (no PyTorch or TensorFlow) that reads a chess position (FEN) and predicts the game state: 91.76% accuracy and 90.25% macro F1 on 72,301 unseen positions. Includes Adam, dropout, L2, early stopping, data augmentation, benchmarks and gradient-checked unit tests. Documentation below is in French.

Un réseau de neurones écrit **à la main** (sans PyTorch ni TensorFlow, avec seulement NumPy pour les calculs sur les tableaux), qui lit une position d'échecs au format FEN et prédit l'état de la partie :

`Nothing` · `Check White` · `Check Black` · `Checkmate White` · `Checkmate Black`

**Réseau rendu : `my_torch_network.nn`, avec 91.76 % de précision et 90.25 % de F1 macro sur 72 301 positions de test jamais vues.**

Projet Epitech (G-CNA-501) réalisé en binôme, avec Milo Bousquet comme lead.
Pour une démonstration guidée sans entraînement : `./demo.sh` (écrit uniquement dans `demo_live/`).

**Contrôle de fuite.** Le jeu de test est séparé avant l'augmentation, mais une
même position peut apparaître dans plusieurs fichiers sources. 1 560 positions de test
(2,2 %) se retrouvent dans l'entraînement, telles quelles ou par symétrie. Sans elles,
le réseau garde **91,65 % de précision et 90,10 % de F1 macro** sur 70 741 positions :
le score annoncé ne vient pas de la mémorisation.

Tests : `make tests_run` (gradients vérifiés par différences finies, softmax, dropout,
Adam, sauvegarde, encodage FEN, labels et augmentation).

## Installation

Il faut Python 3 et NumPy. Matplotlib n'est nécessaire que pour tracer les benchmarks.

```sh
pip3 install numpy matplotlib
make            # crée ./my_torch_analyzer et ./my_torch_generator (règles : all, clean, fclean, re)
```

## Utilisation

```sh
# Prédire l'état de chaque échiquier du fichier (une ligne par position, dans l'ordre)
./my_torch_analyzer --predict my_torch_network.nn chessboards.txt

# Entraîner un réseau (le fichier contient "FEN label" sur chaque ligne)
./my_torch_analyzer --train --save trained.nn my_torch_network.nn train.txt

# Générer des réseaux vierges à partir d'une configuration
./my_torch_generator configs/my_torch_network.conf 3   # -> my_torch_network_1.nn ... _3.nn
```

En cas d'erreur, le message part sur la sortie d'erreur et le programme s'arrête avec le code 84.

```
$> ./my_torch_analyzer --predict my_torch_network.nn chessboards.txt
Checkmate Black
Nothing
Nothing
Check White
Nothing
```

## Le réseau

| | |
|---|---|
| Entrée | 769 valeurs : 12 plans one-hot de 8×8 (un par type de pièce) + le trait |
| Architecture | 769 → 512 → 256 → 5, ReLU, dropout 0.2 |
| Sortie | softmax sur 5 classes, loss cross-entropy |
| Apprentissage | Adam (lr 0.001), batch 256, L2 0.0001, early stopping (patience 8) |
| Données | 650 718 positions ×4 par augmentation (miroir et échange des couleurs) |

Chaque choix a été mesuré contre des alternatives : voir [doc/hyperparametres.md](doc/hyperparametres.md).

## Le module `my_torch` (`src/mlp/`)

C'est un module générique, qui ne sait rien des échecs :
- **Couches** : `Dense` (initialisation He, Xavier ou LeCun), `Dropout`, `ReLU`, `Tanh`, `Sigmoid`
- **Loss** : softmax + cross-entropy
- **Optimiseurs** : `SGD`, `Momentum`, `Adam`, tous avec weight decay L2
- **Entraînement** : mini-batchs mélangés, validation, early stopping avec restauration des meilleurs poids, lr_decay, historique
- **Prédiction** : classique ou MC dropout
- **Métriques** : accuracy, precision, recall, F1, F1 macro, matrice de confusion
- **Création / sauvegarde / chargement** : `Network.from_config`, `save`, `load` (fichier `.nn`), et `my_torch_generator`

```python
from src.mlp.config import load_config
from src.mlp.network import Network
from src.mlp.trainer import train

net = Network.from_config(load_config("configs/my_torch_network.conf"))
history = train(net, inputs, targets, net.config["training"])
net.save("mon_reseau.nn")
```

## Reproduire l'entraînement

```sh
make
python3 -m src.clean.build_dataset                                  # dataset/ -> train.txt (90 %) + test.txt (10 %)
python3 -m src.clean.augment dataset/train.txt dataset/train_aug.txt  # augmentation ×4
./my_torch_generator configs/my_torch_network.conf 1
./my_torch_analyzer --train --save my_torch_network.nn my_torch_network_1.nn dataset/train_aug.txt
python3 -m src.Graphique.evaluate my_torch_network.nn dataset/test.txt doc/benchmarks/final
```

Benchmarks : `python3 -m src.Graphique.benchmark <campagne> configs/benchmarks/<campagne>/*.conf`

## Documentation

| Document | Contenu |
|---|---|
| [doc/theorie.md](doc/theorie.md) | définitions exactes des notions et technologies utilisées |
| [doc/architecture.md](doc/architecture.md) | organisation du code, formats `.conf` et `.nn`, interface des couches |
| [doc/hyperparametres.md](doc/hyperparametres.md) | historique de tous les essais, résultats et justification de chaque choix |
| [doc/benchmarks.md](doc/benchmarks.md) | graphiques : courbes d'apprentissage, scores, matrices de confusion, parallélisme |

## Organisation

```
src/mlp/        module de réseau de neurones (générique)
src/clean/      préparation des données d'échecs (FEN, labels, dataset, augmentation)
src/Graphique/  benchmarks et graphiques
src/analyzer.py my_torch_analyzer
configs/        configuration du réseau rendu et des campagnes de benchmark
doc/            documentation et résultats
```
