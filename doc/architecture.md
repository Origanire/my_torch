# Architecture du projet

## Organisation

```
my_torch_analyzer, my_torch_generator   exécutables (générés par make depuis src/bin/)
Makefile                                all, clean, fclean, re
my_torch_network.nn                     réseau pré-entraîné rendu
configs/                                fichiers .conf (réseau rendu + campagnes de benchmark)
src/
  mlp/          module de réseau de neurones GÉNÉRIQUE, il ne connaît rien aux échecs
    layers.py       Dense, Dropout
    activations.py  ReLU, Tanh, Sigmoid
    losses.py       softmax, SoftmaxCrossEntropy
    optimizers.py   SGD, Momentum, Adam (+ weight decay L2)
    network.py      Network : forward, backward, entraînement par mini-batchs, prédiction, save/load
    trainer.py      boucle d'entraînement : validation, early stopping, lr_decay, historique
    config.py       lecture et validation des .conf
    metrics.py      accuracy, precision, recall, F1, matrice de confusion
    generator.py    my_torch_generator
  clean/        tout ce qui est spécifique aux échecs : préparation des données
    fen.py          FEN -> vecteur de 769 valeurs
    labels.py       label texte <-> numéro de classe
    dataset.py      lecture d'un fichier d'échiquiers (en parallèle), cache .npy
    build_dataset.py  fusion de dataset/ et séparation train/test
  Graphique/    benchmarks et graphiques
    benchmark.py    entraîne plusieurs .conf en parallèle et trace les comparaisons
    bench_parsing.py  mesure le gain du parallélisme sur la lecture des FEN
    plots.py        courbes d'apprentissage, scores, matrice de confusion
  analyzer.py   my_torch_analyzer : relie clean/ (les données) et mlp/ (le réseau)
doc/            documentation et résultats des benchmarks
```

**Pourquoi séparer `mlp/` et `clean/` ?** Le sujet demande un module de réseau **générique**. `src/mlp/` n'importe rien de `src/clean/` : on pourrait l'utiliser tel quel pour classer des images ou des spirales, ce qu'on a fait pendant les tests. Les tailles d'entrée (769) et de sortie (5) ne sont que des valeurs de configuration. Le seul endroit qui relie les deux parties est `src/analyzer.py`, qui vérifie que le réseau chargé a bien ces dimensions.

## Interface commune des couches

Chaque couche (Dense, ReLU, Tanh, Sigmoid, Dropout) expose trois méthodes :
- `forward(X)` : calcule la sortie et garde ce qui sera utile au backward ;
- `backward(dY)` : calcule les gradients de ses paramètres et renvoie `dX` ;
- `params()` : renvoie la liste des paires `(paramètre, gradient)`, vide pour une activation.

Grâce à cette interface, `Network` n'est qu'une liste de couches. Ajouter un nouveau type de couche ne demande aucune modification du réseau ni des optimiseurs.

## Fichier de configuration (`.conf`)

Le format est **INI**, lu par `configparser`, qui fait partie de la bibliothèque standard. Il est lisible et modifiable à la main, et on peut y mettre des commentaires. Toutes les clés sont facultatives : les valeurs absentes prennent les défauts de [src/mlp/config.py](../src/mlp/config.py). Une clé inconnue ou une valeur invalide provoque une erreur explicite.

```ini
[network]                  ; hyperparamètres du modèle
input_size = 769           ; taille de l'entrée
output_size = 5            ; nombre de classes
hidden_layers = 256, 128   ; tailles des couches cachées (vide = modèle linéaire)
activation = relu          ; relu | tanh | sigmoid
init = he                  ; he | xavier | lecun
dropout = 0.0              ; probabilité d'éteindre un neurone caché, dans [0, 1[

[training]                 ; hyperparamètres de l'algorithme
optimizer = adam           ; sgd | momentum | adam
learning_rate = 0.001
batch_size = 256
epochs = 20                ; maximum (l'early stopping peut arrêter avant)
weight_decay = 0.0         ; coefficient de régularisation L2
lr_decay = 1.0             ; learning rate multiplié par ce facteur après chaque epoch
validation_split = 0.1     ; part des exemples gardée pour la validation
patience = 5               ; epochs sans amélioration avant l'early stopping
seed = 42                  ; graine aléatoire (reproductibilité)
```

## Fichier de sauvegarde (`.nn`)

**Ce qu'il faut sauvegarder, et seulement ça :**
| Contenu | Pourquoi |
|---|---|
| `format_version` | pour refuser proprement un fichier d'une ancienne version |
| `config` (JSON) | pour reconstruire les mêmes couches et continuer l'entraînement avec les mêmes hyperparamètres |
| `weights_i`, `bias_i` pour chaque couche dense | ce sont les seuls paramètres appris |
| `metadata` (JSON) | traçabilité : nombre d'epochs effectuées, meilleure loss et meilleure précision de validation |

**Ce qu'on ne sauvegarde pas :** les activations et le dropout, qui n'ont pas de paramètres et sont recréés à partir de la config, ni les moments d'Adam, puisque chaque `--train` démarre avec un optimiseur neuf.

**Le format :** une archive **NPZ** (un ZIP de tableaux `.npy`, compressé), écrite avec `np.savez_compressed` et relue avec `np.load(allow_pickle=False)`.
- **Il est exact** : les poids sont stockés en binaire `float32`, sans aucun arrondi. Une écriture décimale en texte arrondirait les valeurs, et le réseau rechargé ne prédirait pas exactement la même chose. On l'a vérifié : les prédictions sont identiques avant et après le rechargement.
- **Il est compact** : 230 661 paramètres en `float32` donnent environ 0,9 Mo, et moins après compression, contre plusieurs Mo en texte.
- **Il est sûr** : `allow_pickle=False` interdit l'exécution de code contenu dans le fichier.
- **Il est autosuffisant** : le fichier seul suffit à recréer le réseau, sans `.conf` à côté.

## Générateur

```
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
```
Pour chaque couple, il crée `nb` réseaux aux poids aléatoires, nommés `<nom du .conf>_<i>.nn`. Le réseau `i` utilise la graine `seed + i` : les réseaux sont différents entre eux, mais reproductibles.

## Analyzer

```
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```
- `--predict` : affiche une prédiction par ligne, dans l'ordre du fichier. Le label attendu peut être présent ou absent.
- `--train` : entraîne le réseau avec les hyperparamètres `[training]` de sa config (validation, early stopping), puis le sauvegarde dans `SAVEFILE`, ou dans `LOADFILE` si `--save` n'est pas donné.
- Les labels sans couleur (`Check`, `Checkmate`) sont acceptés : la couleur se déduit du trait.
- En `--predict`, le label attendu est facultatif et n'est pas lu : un label inconnu (par exemple `Stalemate`) ne bloque pas la prédiction. En `--train`, il est obligatoire et doit être valide.
- En cas d'erreur (arguments, fichier illisible, FEN invalide avec son numéro de ligne, label inconnu, réseau de mauvaises dimensions), un message part sur la sortie d'erreur et le programme s'arrête avec le code 84.

## Reproduire l'entraînement

```sh
make
python3 -m src.clean.build_dataset                 # dataset/ -> dataset/train.txt + dataset/test.txt
./my_torch_generator configs/my_torch_network.conf 1
./my_torch_analyzer --train --save my_torch_network.nn my_torch_network_1.nn dataset/train.txt
./my_torch_analyzer --predict my_torch_network.nn dataset/test.txt
python3 -m src.Graphique.benchmark <campagne> configs/benchmarks/<campagne>/*.conf   # benchmarks
```
