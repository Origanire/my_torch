# Historique des hyperparamètres et justification des choix

Chaque choix du réseau final a été **mesuré** contre des alternatives. Ce document garde l'historique de tous les essais, dans l'ordre où ils ont été faits, avec leurs résultats et la décision prise. Les définitions des notions utilisées sont dans [theorie.md](theorie.md), et les graphiques dans [benchmarks.md](benchmarks.md).

## Protocole

- **Données** : les 723 019 positions de `dataset/`, mélangées avec une graine fixe (42), puis séparées en **train** (650 718 positions, 90 %) et **test** (72 301 positions, 10 %) par [src/clean/build_dataset.py](../src/clean/build_dataset.py). L'entraînement garde encore 10 % de son train pour la **validation**. Il reste donc 585 647 positions pour apprendre.
- **Mesure** : tous les scores sont calculés sur le jeu de **test**, que le réseau ne voit jamais pendant l'entraînement. On indique la précision globale (accuracy) et le **F1 macro**, la moyenne des F1 des 5 classes, qui n'est pas dominée par la classe `Nothing` (41 % des données).
- **Méthode** : **un seul facteur à la fois**. On part d'une configuration de référence et chaque campagne ne change qu'un hyperparamètre. Tout le reste est identique, y compris la graine, donc les mêmes poids initiaux et le même mélange. La différence de score est ainsi attribuable à ce seul choix.
- **Arrêt** : l'early stopping avec `patience = 5` arrête l'entraînement et garde la meilleure epoch selon la loss de validation.
- **Reproduire** : `python3 -m src.Graphique.benchmark <campagne> configs/benchmarks/<campagne>/*.conf`
- **Machine** : macOS, 15 cœurs, 24 Go de RAM. 4 entraînements tournent en parallèle, avec 3 threads de calcul matriciel chacun. Les temps sont donc à comparer **entre eux** à l'intérieur d'une campagne.

## Configuration de référence (v0)

Fichier : [configs/benchmarks/baseline/baseline.conf](../configs/benchmarks/baseline/baseline.conf)

| Hyperparamètre | Valeur | Justification a priori |
|---|---|---|
| Entrée | 769 (12×64 one-hot + trait) | un plan par type de pièce, aucune relation d'ordre artificielle ; le trait détermine la couleur du label |
| Couches cachées | 256, 128 | taille moyenne : assez de capacité pour des motifs d'attaque, entraînement rapide |
| Activation | ReLU | dérivée de 1 pour x > 0, donc pas d'écrasement du gradient ; calcul trivial |
| Initialisation | He | conçue pour ReLU |
| Sortie | 5, softmax + cross-entropy | classification à 5 classes exclusives |
| Optimiseur | Adam, lr = 0.001 | valeurs par défaut de l'article d'origine, robustes |
| Batch | 256 | compromis entre le bruit du gradient et la vitesse (produits matriciels efficaces) |
| Régularisation | aucune | référence |
| Epochs | 30 max, patience 5 | |

**Résultat v0 : 83.27 % de précision, 81.30 % de F1 macro sur le test.** La meilleure epoch est la 5ᵉ, et l'arrêt intervient à la 10ᵉ.

**Diagnostic : surapprentissage net.** À partir de l'epoch 5, la loss de validation remonte (0.404 → 0.463) alors que la loss d'entraînement continue de baisser (0.29 → 0.18). La précision d'entraînement atteint 95 %, contre 84 % en validation.

---

## Série 1 : un facteur à la fois

### Optimiseur
| Variante | Test acc | F1 macro | Meilleure epoch |
|---|---|---|---|
| SGD, lr = 0.05 | 81.14 % | 78.98 % | 29/30 |
| Momentum (β = 0.9), lr = 0.01 | 81.96 % | 79.77 % | 18/23 |
| **Adam, lr = 0.001** | **83.27 %** | **81.30 %** | **5/10** |

Adam atteint le meilleur score en 5 epochs, contre 18 pour Momentum. SGD n'a toujours pas convergé au bout de 30 epochs. Le pas adaptatif d'Adam compte beaucoup ici : les entrées sont creuses (environ 15 cases occupées sur 768), et les poids des cases rarement occupées reçoivent rarement un gradient. Adam leur donne un pas plus grand. **Choix : Adam.**

### Learning rate (Adam)
| lr | Test acc | F1 macro | Meilleure epoch |
|---|---|---|---|
| 0.0001 | 80.47 % | 78.47 % | 22/27 |
| 0.0003 | 81.64 % | 79.64 % | 9/14 |
| 0.001 | 83.27 % | 81.30 % | 5/10 |
| 0.003 | 84.49 % | 82.47 % | 4/9 |
| **0.01** | **85.18 %** | **83.18 %** | 6/11 |

Plus le learning rate est grand, meilleur est le score, dans cette plage. Les grands pas ajoutent du bruit dans l'optimisation, ce qui agit comme une régularisation et retarde le surapprentissage. On garde ce résultat en tête pour la série 2, où il sera testé **combiné** à la régularisation : ce bruit pourrait devenir inutile, voire nuisible, quand le dropout régularise déjà.

### Taille de batch
| Batch | Test acc | F1 macro | Temps par epoch |
|---|---|---|---|
| 32 | 84.79 % | 82.75 % | 8.9 s |
| 64 | 84.69 % | 82.75 % | 5.0 s |
| 256 | 83.27 % | 81.30 % | 3.1 s |
| 1024 | 81.81 % | 79.68 % | 2.2 s |
| 4096 | 81.72 % | 79.61 % | 1.9 s |

Un petit batch donne un gradient plus bruité, et ce bruit régularise, pour le même effet que le grand learning rate. Mais un batch de 32 coûte 3 fois plus de temps par epoch qu'un batch de 256 pour seulement 1,5 point de plus. **Le batch de 64 est le meilleur compromis**, testé en série 2.

### Architecture (hyperparamètres du modèle)
| Couches cachées | Paramètres | Test acc | F1 macro | Écart train-val | Temps par epoch |
|---|---|---|---|---|---|
| aucune (linéaire) | 3 850 | 64.05 % | 60.87 % | -0.3 pt | 0.4 s |
| 128 | 99 205 | 82.78 % | 80.73 % | 6.0 pts | 1.5 s |
| **256, 128** | **230 661** | **83.27 %** | **81.30 %** | **7.5 pts** | **2.8 s** |
| 512, 256 | 526 853 | 83.71 % | 81.62 % | 9.9 pts | 5.8 s |
| 1024, 512, 256 | 1 445 893 | 83.75 % | 81.54 % | 9.9 pts | 13.4 s |
| 512 × 4 | 1 184 773 | 84.08 % | 82.04 % | 9.7 pts | 10.2 s |

- **Le modèle linéaire plafonne à 64 %.** C'est la preuve que le problème n'est pas linéairement séparable. Savoir si un roi est attaqué demande de combiner « telle pièce est sur telle case » **ET** « le chemin entre elle et le roi est libre », et une somme pondérée ne peut pas exprimer un « ET ». Il faut des couches cachées.
- **Au-delà de 256-128, la capacité ne sert presque plus** (+0,8 point au mieux, pour 4 à 5 fois plus de calcul). Les grands réseaux surapprennent davantage (écart train-val de 10 points contre 7,5). Le facteur limitant est la **généralisation**, pas la capacité. **Choix : 256-128, et l'effort porte sur la régularisation.** Le 512-256 sera retesté une fois régularisé.

### Activation
| Activation | Test acc | F1 macro | Meilleure epoch |
|---|---|---|---|
| **ReLU** | 83.27 % | 81.30 % | **5/10** |
| Tanh | 82.54 % | 80.67 % | 11/16 |
| Sigmoïde | 83.44 % | 81.54 % | 24/29 |

Les scores sont proches, mais la sigmoïde a besoin de **5 fois plus d'epochs** pour y arriver. Sa dérivée est au maximum de 0,25, donc le gradient s'affaiblit à chaque couche (*vanishing gradient*). **Choix : ReLU**, qui donne le même score bien plus vite.

### Initialisation
| Init | Test acc | F1 macro |
|---|---|---|
| He | 83.27 % | 81.30 % |
| LeCun | 83.62 % | 81.66 % |
| Xavier | 83.72 % | 81.74 % |

Les écarts (moins de 0,5 point) sont de l'ordre de la variation due au hasard. Avec seulement 2 couches cachées, la variance des activations n'a pas le temps de dériver, quelle que soit l'initialisation. **Choix : He**, qui reste la référence théorique pour ReLU. Ce paramètre n'est pas déterminant ici.

### Régularisation (contre le surapprentissage)
| Variante | Test acc | F1 macro | Meilleure epoch | Écart train-val |
|---|---|---|---|---|
| aucune | 83.27 % | 81.30 % | 5/10 | 7.5 pts |
| L2 λ = 0.001 | 82.29 % | 79.99 % | 20/25 | 2.2 pts |
| L2 λ = 0.0001 | 86.02 % | 84.01 % | 13/18 | 8.4 pts |
| dropout 0.4 | 86.78 % | 85.21 % | 30/30 | 6.1 pts |
| dropout 0.2 | 87.51 % | 85.75 % | 29/30 | 8.5 pts |
| **dropout 0.2 + L2 0.0001** | **88.09 %** | **86.38 %** | **30/30** | **4.8 pts** |

**C'est le facteur le plus important : +4,8 points.**
- Une L2 trop forte (0.001) empêche le réseau d'apprendre (sous-apprentissage). À 0.0001, elle aide.
- Le dropout 0.2 fait mieux que 0.4, qui éteint trop de neurones et ralentit l'apprentissage.
- L'écart train-val est mesuré à la meilleure epoch, qui n'est pas la même selon les variantes. Avec le dropout seul, l'écart reste de 8,5 points, mais il est atteint à l'epoch 29 au lieu de la 5. Le dropout ne supprime pas l'écart : il permet de **s'entraîner bien plus longtemps avant que la validation se dégrade**, et on gagne ainsi 4 points de test. Une L2 de 0.001 donne un petit écart (2,2 points) parce que le réseau apprend mal, aussi bien en entraînement qu'en validation : c'est du sous-apprentissage, pas une bonne généralisation.
- Avec dropout 0.2 + L2 0.0001, l'écart passe à 4,8 points. **La loss de validation baissait encore à l'epoch 30**, la limite : le réseau n'avait pas fini d'apprendre.

**Choix : dropout 0.2 + L2 0.0001**, avec plus d'epochs en série 2.

### Décroissance du learning rate
| lr_decay | Test acc | F1 macro |
|---|---|---|
| **1.0 (aucune)** | **83.27 %** | **81.30 %** |
| 0.9 | 83.12 % | 81.14 % |
| 0.8 | 82.74 % | 80.77 % |

Ça n'apporte rien : l'early stopping arrête l'entraînement avant que la décroissance ait un effet utile, et Adam adapte déjà ses pas. **Choix : pas de décroissance.**

### Analyse des erreurs (matrice de confusion, dropout 0.2 + L2)
- **Aucune confusion de couleur.** White↔Black vaut 0 sur 72 301 positions. Le trait en entrée suffit, ce qui confirme le choix de l'encodage.
- **Erreurs principales : Check ↔ Checkmate.** Par exemple, 1 192 « Check White » sont prédits « Checkmate White ». Pour distinguer un échec d'un mat, il faut vérifier qu'**aucun** coup ne sauve le roi (fuite, prise de l'attaquant, interposition). C'est un raisonnement sur tous les coups possibles, bien plus difficile pour un MLP que de détecter une attaque.
- Ensuite viennent **Nothing ↔ Check** (876 + 782 échecs pris pour « Nothing », 1 019 + 967 « Nothing » pris pour des échecs). Une attaque à distance (fou, tour, dame) dépend de toutes les cases entre l'attaquant et le roi.

---

## Série 2 : combinaisons

La série 1 mesure chaque facteur **isolément**. Mais deux gains ne s'additionnent pas forcément : deux choix qui corrigent le même défaut peuvent faire doublon. On repart donc de la meilleure régularisation (dropout 0.2 + L2 0.0001), avec jusqu'à 60 epochs et `patience = 6` puisqu'elle progressait encore à l'epoch 30, et on y ajoute un à un les autres gains de la série 1.

Fichiers : [configs/benchmarks/combination/](../configs/benchmarks/combination/).

| Variante (base : dropout 0.2 + L2 0.0001) | Test acc | F1 macro | Meilleure epoch |
|---|---|---|---|
| **base, 60 epochs max** | **88.77 %** | **87.12 %** | 45/51 |
| + lr 0.003 | 85.82 % | 83.82 % | 29/35 |
| + batch 64 | 87.68 % | 85.94 % | 54/60 |
| + couches 512, 256 | 89.45 % | 87.67 % | 59/60 |

- **Plus d'epochs** : 88.09 % → 88.77 %. La meilleure epoch est la 45ᵉ, ce qui confirme que la limite de 30 epochs coupait l'apprentissage.
- **Le grand learning rate et le petit batch ne servent plus** : ils font même perdre 3 et 1,1 points. En série 1, ils aidaient parce que leur bruit régularisait. Maintenant que le dropout et la L2 régularisent, ce bruit en plus ne fait qu'empêcher une convergence fine. **On garde lr = 0.001 et batch = 256.**
- **Le réseau plus large devient utile** : +0,7 point. En série 1, sans régularisation, sa capacité supplémentaire servait à surapprendre. Régularisée, elle sert à apprendre davantage de motifs. Il progressait encore à l'epoch 59/60. **On garde 512, 256.**

### Encodage de l'entrée (proposition alternative)

Alternative à l'encodage one-hot : une **valeur par case** (pion 1, cavalier 2, fou 3, tour 4, dame 5, roi 6, en négatif pour les noirs, 0 si la case est vide), plus le trait. Il ne fait que 65 valeurs au lieu de 769, soit 12 fois moins. Script : [src/Graphique/bench_encoding.py](../src/Graphique/bench_encoding.py). Même configuration (dropout 0.2 + L2, 256-128) et même graine.

| Encodage | Entrées | Test acc | F1 macro | Temps |
|---|---|---|---|---|
| **one-hot** | 769 | **88.77 %** | **87.12 %** | 158 s |
| valeurs de pièces | 65 | 70.16 % | 68.77 % | 76 s |

**−18,6 points avec l'encodage compact**, ce qui confirme la justification théorique. Avec une seule valeur par case, le premier neurone calcule `w × valeur`. Pour lui, une dame (5) est « 5 fois un pion », un cavalier se situe entre un pion et un fou, et une pièce noire est l'opposé d'une pièce blanche. Ces relations n'ont aucun sens pour la question « cette pièce attaque-t-elle le roi ? ». Le réseau doit d'abord dépenser de la capacité pour « défaire » ce codage avant de pouvoir apprendre la règle. En one-hot, chaque couple (pièce, case) a son propre poids, ce qui permet d'apprendre directement des motifs comme « tour en a6 ». **Choix : one-hot.**

### Augmentation de données (proposition alternative)

Idée : certaines transformations de l'échiquier ne changent pas l'état de la partie, ou le changent de façon connue. Elles fabriquent de nouveaux exemples valides ([src/clean/augment.py](../src/clean/augment.py)) :
- **miroir gauche-droite** (colonne a ↔ colonne h) : les attaques sont conservées, le label ne change pas ;
- **échange des couleurs** (on retourne le plateau, on échange les pièces blanches et noires, le trait s'inverse) : White ↔ Black dans le label.

En combinant les deux, le jeu d'entraînement passe de 650 718 à **2 602 872** positions. **Vérification** : on a recalculé l'état réel de 15 000 positions transformées avec un moteur de règles indépendant (`python-chess`, utilisé uniquement pour ce contrôle et pas dans le projet). Résultat : 0 erreur, et 0 erreur aussi sur 5 000 labels d'origine du dataset.

| Données (base : dropout 0.2 + L2, 256-128) | Test acc | F1 macro | Meilleure epoch | Temps |
|---|---|---|---|---|
| originales (650 718) | 88.77 % | 87.12 % | 45/51 | 219 s |
| **augmentées ×4 (2 602 872)** | **90.42 %** | **88.89 %** | 38/44 | 522 s |

**+1,65 point**, le meilleur gain de la série 2. Plus d'exemples variés, c'est la façon la plus directe de réduire le surapprentissage. Le réseau ne peut pas mémoriser une position quand il en voit aussi le miroir et la version aux couleurs inversées, et il doit apprendre la règle.

Point de vigilance : la validation est tirée du jeu **augmenté**. Elle peut donc contenir une variante d'une position d'entraînement, ce qui risquerait de la rendre optimiste. Le jeu de **test**, lui, n'est pas augmenté et provient de positions indépendantes. Comme le score de test (90.42 %) est cohérent avec celui de validation (90.25 %), ce biais est négligeable en pratique.

---

## Réseau final : `my_torch_network.nn`

Fichier : [configs/my_torch_network.conf](../configs/my_torch_network.conf). Entraînement avec le vrai pipeline :

```sh
./my_torch_generator configs/my_torch_network.conf 1
./my_torch_analyzer --train --save my_torch_network.nn my_torch_network_1.nn dataset/train_aug.txt
```

| Hyperparamètre | Valeur | Justification mesurée |
|---|---|---|
| Encodage | one-hot 769 | +18,6 pts par rapport à l'encodage par valeurs de pièces |
| Couches cachées | 512, 256 | +0,7 pt par rapport à 256-128 une fois régularisé ; le gain des réseaux plus grands est inférieur à 1 pt en série 1 pour 2 à 5 fois plus de calcul |
| Activation | ReLU | même score que la sigmoïde, mais 5 fois moins d'epochs |
| Initialisation | He | référence théorique pour ReLU ; écarts mesurés < 0,5 pt |
| Dropout | 0.2 | +4,2 pts ; 0.4 fait moins bien |
| Weight decay (L2) | 0.0001 | +0,6 pt combiné au dropout ; 0.001 provoque du sous-apprentissage |
| Optimiseur | Adam | +1,3 pt par rapport à Momentum, et 3 à 6 fois moins d'epochs |
| Learning rate | 0.001 | 0.003 fait perdre 3 pts une fois régularisé |
| Batch | 256 | 64 fait perdre 1,1 pt une fois régularisé, en 2,2 fois plus de temps |
| lr_decay | 1.0 | la décroissance n'apporte rien |
| Epochs / patience | 100 / 8 | l'early stopping choisit l'epoch ; une grande patience laisse le temps aux petites améliorations sur 2,3 M d'exemples |
| Données | augmentées ×4 | +1,65 pt |

**Déroulement** : 2 342 585 exemples d'entraînement et 260 287 de validation. L'early stopping arrête l'entraînement à l'epoch 55, et la meilleure est la 47ᵉ (val_loss 0.2174). Une epoch prend environ 20 s.

**Résultat sur le jeu de test (72 301 positions jamais vues) : 91.76 % de précision, 90.25 % de F1 macro**, contre 83.27 % et 81.30 % pour la configuration de référence v0.

| Classe | Precision | Recall | F1 |
|---|---|---|---|
| Nothing | 96.59 % | 95.56 % | 96.07 % |
| Check White | 88.60 % | 87.87 % | 88.23 % |
| Check Black | 89.49 % | 87.93 % | 88.70 % |
| Checkmate White | 87.29 % | 89.70 % | 88.48 % |
| Checkmate Black | 87.77 % | 91.85 % | 89.76 % |

**Surapprentissage maîtrisé** : la loss de validation se stabilise sans remonter, et l'écart de précision entre entraînement et validation est inférieur à 1 point (contre 7,5 points pour v0). La loss d'entraînement affichée est **plus haute** que celle de validation parce qu'elle est mesurée pendant l'entraînement, avec 20 % des neurones éteints, alors que la validation utilise le réseau complet.

### MC dropout en prédiction (alternative écartée)
| Prédiction | Test acc | F1 macro | Temps (72 301 positions) |
|---|---|---|---|
| **classique (dropout coupé)** | **91.76 %** | **90.25 %** | **0.1 s** |
| MC dropout, 10 passes | 91.57 % | 90.04 % | 2.0 s |

Le MC dropout est 20 fois plus lent, pour un score légèrement inférieur. Faire la moyenne de plusieurs passes réduit la variance d'un modèle instable, mais notre modèle est déjà bien régularisé et entraîné sur beaucoup de données. **Choix : prédiction classique.** L'option reste disponible dans le module : `Network.predict(inputs, mc_samples=T)`.

---

## Pistes non retenues, et pourquoi

| Piste | Raison |
|---|---|
| Sortie à 3 classes (Nothing/Check/Checkmate) + couleur déduite par une règle | La couleur n'est jamais confondue avec 5 sorties (0 erreur sur 72 301), donc la règle n'apporterait rien, et le sujet impose une solution par apprentissage. |
| Réseau convolutif (CNN) | Il exploiterait la géométrie 8×8 et pourrait mieux détecter les lignes d'attaque, mais ce serait une nouvelle famille de couches à écrire et à dériver à la main. C'est la piste la plus prometteuse pour aller au-delà de 92 %. |
| Ajouter des cases attaquées calculées à la main en entrée | Le réseau ne ferait que recopier une règle codée à la main, ce qui n'est plus vraiment de l'apprentissage. |
| Recherche aléatoire ou métaheuristique sur toutes les combinaisons | Le budget est trop grand pour un gain incertain. La recherche un facteur à la fois, suivie d'une série de combinaisons, isole l'effet de chaque choix, et c'est ce qu'il faut pouvoir justifier. |
