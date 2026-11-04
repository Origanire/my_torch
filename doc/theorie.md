# Théorie et technologies utilisées

Ce document définit chaque notion utilisée dans le projet, et indique où elle apparaît dans le code.

Notations : `X` est la matrice des entrées d'un batch (une ligne par exemple), `N` la taille du batch, `W` une matrice de poids, `b` un vecteur de biais et `L` la loss. `dW` désigne `∂L/∂W`, la dérivée partielle de la loss par rapport à `W`.

---

## 1. Le problème

### Notation FEN (Forsyth–Edwards Notation)
Format texte standard qui décrit une position d'échecs sur une ligne, en 6 champs séparés par des espaces :
1. **Le placement des pièces**, rangée par rangée de la 8ᵉ à la 1ʳᵉ, séparées par `/`. Dans chaque rangée, on lit de la colonne a à la colonne h. Une lettre désigne une pièce : majuscule pour les blancs, minuscule pour les noirs (`P` pion, `N` cavalier, `B` fou, `R` tour, `Q` dame, `K` roi). Un chiffre `n` désigne `n` cases vides consécutives.
2. **Le trait** : `w` si c'est aux blancs de jouer, `b` si c'est aux noirs.
3. **Les roques encore possibles** (`KQkq`, ou `-` s'il n'y en a plus).
4. **La case de prise en passant** (`-` si aucune).
5. **Le compteur de demi-coups** depuis la dernière prise ou le dernier coup de pion (sert à la règle des 50 coups).
6. **Le numéro du coup.**

### Échec, mat, pat
- **Échec** : le roi du camp qui a le trait est attaqué par au moins une pièce adverse.
- **Échec et mat** : le roi est en échec et aucun coup légal ne permet de le sortir de l'échec. La partie est perdue pour le camp qui a le trait.
- **Pat** : le camp qui a le trait n'est **pas** en échec mais n'a aucun coup légal. La partie est nulle. C'est un bonus du sujet, et le dataset n'en contient aucun.

Le camp qui subit l'échec est toujours celui qui a le trait. Le label « Check White » (« les blancs font échec ») va donc toujours avec le trait `b`. On l'a vérifié sur les 723 019 positions du dataset.

### Apprentissage supervisé
On apprend une fonction `f : entrée → sortie` à partir d'exemples **étiquetés** `(xᵢ, yᵢ)`, dont on connaît la bonne réponse. L'apprentissage consiste à ajuster les paramètres de `f` pour réduire l'écart entre `f(xᵢ)` et `yᵢ`.

### Classification multiclasse
C'est un apprentissage supervisé où la sortie est une catégorie parmi `C` possibles. Ici `C = 5` : `Nothing`, `Check White`, `Check Black`, `Checkmate White`, `Checkmate Black` ([src/clean/labels.py](../src/clean/labels.py)).

---

## 2. Représentation des données

### Encodage one-hot
On représente une variable catégorielle à `k` valeurs par un vecteur de `k` composantes : un seul `1` à la position de la valeur, et des `0` partout ailleurs. On évite ainsi d'introduire une relation d'ordre artificielle entre les catégories (pion = 1 < cavalier = 2 < …), relation que le réseau, qui ne fait que des sommes pondérées, interpréterait comme une grandeur.

### Encodage de la position (769 entrées)
Code : [src/clean/fen.py](../src/clean/fen.py).
- **768 valeurs** = 12 plans × 64 cases. Le plan `p` correspond au type de pièce `p` (6 types × 2 couleurs). La valeur d'indice `p × 64 + rangée × 8 + colonne` vaut 1 si cette pièce occupe cette case, et 0 sinon.
- **1 valeur** pour le trait : 1 si ce sont les blancs qui jouent, 0 si ce sont les noirs.

Les champs 3 à 6 du FEN sont ignorés, parce qu'ils n'influencent pas l'état échec/mat. Seule exception, extrêmement rare : une prise en passant qui sauverait d'un mat.

### Jeux d'entraînement, de validation et de test
- **Entraînement** : les exemples utilisés pour calculer les gradients et modifier les poids.
- **Validation** : des exemples mis de côté, **jamais** utilisés pour modifier les poids. On les évalue à chaque epoch pour détecter le surapprentissage et piloter l'early stopping.
- **Test** : des exemples mis de côté **dès le départ** ([src/clean/build_dataset.py](../src/clean/build_dataset.py), 10 % du dataset). On ne les utilise que pour la mesure finale des benchmarks. Le score de test estime la performance sur des positions inconnues.

---

## 3. Le réseau de neurones

### Neurone artificiel
Fonction `y = f(Σᵢ wᵢ·xᵢ + b)` : une somme pondérée des entrées `xᵢ` par des poids `wᵢ`, plus un biais `b`, à laquelle on applique une fonction d'activation `f`.

### Perceptron multicouche (MLP, Multi-Layer Perceptron)
Réseau composé de **couches denses** successives séparées par des fonctions d'activation non linéaires. Chaque neurone d'une couche est connecté à toutes les sorties de la couche précédente. Le modèle de ce projet est un MLP ([src/mlp/network.py](../src/mlp/network.py)).

### Couche dense (entièrement connectée)
Transformation affine appliquée à tout un batch d'un coup : `Y = X·W + b`, avec `X` de forme `(N, n_in)`, `W` de forme `(n_in, n_out)` et `b` de forme `(n_out)`. Code : `Dense` dans [src/mlp/layers.py](../src/mlp/layers.py).

### Fonction d'activation
Fonction non linéaire appliquée élément par élément entre deux couches. Sans elle, une suite de couches denses se réduirait à une seule transformation affine (`(X·W₁)·W₂ = X·(W₁·W₂)`), et le réseau ne pourrait séparer les classes que par des hyperplans. Code : [src/mlp/activations.py](../src/mlp/activations.py).

| Nom | Définition | Dérivée |
|---|---|---|
| ReLU (*Rectified Linear Unit*) | `f(x) = max(0, x)` | `1` si `x > 0`, sinon `0` |
| Tanh (tangente hyperbolique) | `f(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)`, valeurs dans ]-1, 1[ | `1 - f(x)²` |
| Sigmoïde | `f(x) = 1 / (1 + e⁻ˣ)`, valeurs dans ]0, 1[ | `f(x)·(1 - f(x))` |

### Softmax
Transforme un vecteur de scores réels `z` (les « logits ») en une distribution de probabilité : `pᵢ = e^{zᵢ} / Σⱼ e^{zⱼ}`. Toutes les composantes sont positives et leur somme vaut 1. Pour éviter les dépassements de capacité, on calcule `softmax(z - max(z))`, qui donne exactement le même résultat. Code : [src/mlp/losses.py](../src/mlp/losses.py).

### Fonction de coût (loss)
Fonction qui mesure l'écart entre la prédiction et la bonne réponse. L'apprentissage cherche à la minimiser.

### Entropie croisée (cross-entropy)
Loss utilisée en classification : `L = -(1/N) Σₙ log(p_{n, yₙ})`, où `p_{n, yₙ}` est la probabilité que le réseau donne à la bonne classe de l'exemple `n`. Elle vaut 0 si le réseau donne une probabilité de 1 à la bonne classe, et elle tend vers l'infini quand cette probabilité tend vers 0. Combinée à la softmax, son gradient par rapport aux logits se simplifie en `∂L/∂z = (p - y) / N`, où `y` est la bonne réponse encodée en one-hot.

---

## 4. L'apprentissage

### Descente de gradient
Méthode itérative de minimisation : à chaque étape, on déplace les paramètres dans la direction opposée au gradient de la loss, `θ ← θ - η·∇L(θ)`. Le gradient indique la direction de plus forte augmentation de la loss.

### Taux d'apprentissage (learning rate, η)
Le coefficient `η` qui fixe la taille de chaque pas. Trop grand, l'optimisation oscille ou diverge. Trop petit, elle converge très lentement ou reste coincée.

### Rétropropagation (backpropagation)
Algorithme qui calcule le gradient de la loss par rapport à tous les paramètres du réseau. On applique la **règle de dérivation en chaîne** couche par couche, de la sortie vers l'entrée. Chaque couche reçoit `dY = ∂L/∂Y` et calcule :
- pour une couche dense : `dW = Xᵀ·dY`, `db = Σₙ dYₙ` et `dX = dY·Wᵀ` ;
- pour une activation : `dX = dY ⊙ f'(X)`, où `⊙` est le produit élément par élément.

Elle transmet ensuite `dX` à la couche précédente. Code : les méthodes `backward` de chaque couche, et `Network.backward`.

### Vérification numérique du gradient (gradient check)
Test de validité de la rétropropagation. On compare le gradient analytique au gradient approché par différence finie centrée, `(L(θ + ε) - L(θ - ε)) / 2ε`. Écart mesuré pendant le développement : de l'ordre de 10⁻¹⁰ en float64, donc les gradients sont corrects.

### Epoch, batch, mini-batch
- **Epoch** : un passage complet sur tous les exemples d'entraînement.
- **Batch** : un sous-ensemble d'exemples traités ensemble pour calculer **un** pas de gradient.
- **Descente de gradient stochastique par mini-batchs (mini-batch SGD)** : à chaque epoch, on mélange les exemples, on les découpe en batchs de taille `B`, puis on fait un pas de gradient par batch. C'est un compromis entre le gradient exact sur tout le dataset (coûteux) et le gradient d'un seul exemple (très bruité). Code : `Network.fit_epoch`.

### Initialisation des poids
On tire les poids initiaux au hasard, selon une loi normale centrée d'écart-type adapté au nombre d'entrées `n_in` de la couche. Le but est que la variance des activations reste stable d'une couche à l'autre : sans ça, les valeurs explosent ou s'écrasent au fil des couches. Si les poids étaient tous identiques, par exemple tous nuls, tous les neurones d'une couche calculeraient la même chose et resteraient identiques pendant tout l'apprentissage.
- **He (Kaiming, 2015)** : `σ = √(2 / n_in)`, adaptée à ReLU, qui annule en moyenne la moitié des valeurs (d'où le facteur 2).
- **Xavier (Glorot & Bengio, 2010)** : `σ = √(2 / (n_in + n_out))`, un compromis qui garde la variance stable dans le forward et dans le backward. Adaptée à tanh et à la sigmoïde.
- **LeCun (1998)** : `σ = √(1 / n_in)`, qui garde la variance stable dans le forward pour des activations symétriques.

Les biais partent à 0. Code : `Dense.__init__`.

### Optimiseurs
Code : [src/mlp/optimizers.py](../src/mlp/optimizers.py).
- **SGD** : `θ ← θ - η·g`, où `g` est le gradient du batch.
- **Momentum** : `v ← β·v + g` puis `θ ← θ - η·v`. La « vitesse » `v` est une moyenne des gradients passés. On accélère dans les directions constantes et on amortit les oscillations. Par défaut, `β = 0.9`.
- **Adam** (*Adaptive Moment Estimation*, Kingma & Ba, 2014) : il tient deux moyennes mobiles exponentielles, `m ← β₁·m + (1-β₁)·g` pour le premier moment (la moyenne des gradients) et `v ← β₂·v + (1-β₂)·g²` pour le second (la moyenne des gradients au carré). On les corrige du biais dû à leur départ à 0 (`m̂ = m / (1-β₁ᵗ)`, `v̂ = v / (1-β₂ᵗ)`), puis on fait `θ ← θ - η·m̂ / (√v̂ + ε)`. Chaque paramètre a ainsi un pas adapté à l'amplitude habituelle de ses gradients. Valeurs utilisées : `β₁ = 0.9`, `β₂ = 0.999`, `ε = 10⁻⁸`.

### Décroissance du learning rate (lr_decay)
Après chaque epoch, `η ← η × lr_decay`, avec `lr_decay` dans ]0, 1]. On fait de grands pas au début, puis des pas plus fins pour converger précisément.

---

## 5. Généralisation et surapprentissage

### Surapprentissage (overfitting) et sous-apprentissage (underfitting)
- **Surapprentissage** : le modèle apprend les particularités, voire le bruit, des exemples d'entraînement au lieu de la règle générale. La loss d'entraînement continue de baisser pendant que la loss de validation stagne ou remonte.
- **Sous-apprentissage** : le modèle est trop simple, ou pas assez entraîné, pour capturer la règle. Les deux losses restent élevées.

### Early stopping (arrêt précoce)
On évalue la loss de validation à chaque epoch. Si elle ne s'est pas améliorée depuis `patience` epochs, on arrête l'entraînement, et on **restaure les poids de la meilleure epoch**. Code : [src/mlp/trainer.py](../src/mlp/trainer.py).

### Régularisation L2 (weight decay)
On ajoute à la loss la pénalité `(λ/2)·Σ W²`. Le gradient de chaque matrice de poids devient alors `g + λ·W`, ce qui ramène les poids vers 0 à chaque pas et défavorise les solutions qui reposent sur quelques poids très grands. Les biais ne sont pas régularisés. Code : `SGD.regularized`.

### Dropout
Pendant l'entraînement, chaque sortie d'une couche cachée est mise à 0 avec la probabilité `p`, tirée à nouveau à chaque batch. Le réseau ne peut plus dépendre d'un neurone précis, et il doit répartir l'information de façon redondante. On utilise le **dropout inversé** : les valeurs conservées sont divisées par `(1 - p)` pendant l'entraînement, ce qui laisse l'espérance inchangée. En prédiction, le dropout est désactivé et aucune correction n'est nécessaire. Code : `Dropout` dans [src/mlp/layers.py](../src/mlp/layers.py).

### MC dropout (Monte Carlo dropout)
En prédiction, on laisse le dropout **actif**, on fait `T` prédictions, puis on fait la moyenne des probabilités obtenues. C'est une approximation d'un ensemble de `T` réseaux, et la dispersion des `T` prédictions estime l'incertitude. Code : `Network.predict_proba(mc_samples=T)`.

---

## 6. Hyperparamètres

### Paramètre et hyperparamètre
- **Paramètre** : une valeur apprise par la descente de gradient (les poids et les biais).
- **Hyperparamètre** : une valeur fixée **avant** l'apprentissage, qui n'est pas modifiée par la descente de gradient.

On distingue deux familles, qui correspondent aux deux sections du fichier `.conf` ([src/mlp/config.py](../src/mlp/config.py)) :
- **hyperparamètres du modèle** `[network]` : ils définissent la fonction calculée (nombre et taille des couches, activation, initialisation, taux de dropout) ;
- **hyperparamètres de l'algorithme** `[training]` : ils définissent la façon d'apprendre (optimiseur, learning rate, taille de batch, nombre d'epochs, weight decay, lr_decay, patience, proportion de validation, graine aléatoire).

### Optimisation des hyperparamètres
Recherche de la combinaison d'hyperparamètres qui maximise un score de validation.
- **Recherche par variation d'un facteur à la fois** (*one-factor-at-a-time*), utilisée dans nos campagnes de benchmark : on part d'une configuration de référence et on ne fait varier qu'un hyperparamètre à la fois. L'effet de ce choix est alors isolé.
- **Recherche en grille** (*grid search*) : on teste toutes les combinaisons d'un ensemble de valeurs.
- **Recherche aléatoire** (*random search*, Bergstra & Bengio, 2012) : on tire des combinaisons au hasard. À budget égal, elle est souvent plus efficace que la grille quand seuls quelques hyperparamètres comptent vraiment.

### Graine aléatoire (seed)
Valeur d'initialisation du générateur de nombres pseudo-aléatoires. Avec la même graine, les mêmes poids initiaux, le même mélange et le même dropout se reproduisent, ce qui rend les expériences reproductibles.

---

## 7. Métriques d'évaluation

Code : [src/mlp/metrics.py](../src/mlp/metrics.py). Pour une classe `c` : `VP` (vrais positifs) = exemples de classe `c` prédits `c`, `FP` (faux positifs) = exemples prédits `c` à tort, `FN` (faux négatifs) = exemples de classe `c` prédits dans une autre classe.

- **Précision globale (accuracy)** : la proportion d'exemples correctement classés.
- **Précision d'une classe (precision)** : `VP / (VP + FP)`, c'est-à-dire la fiabilité d'une prédiction « c ».
- **Rappel (recall)** : `VP / (VP + FN)`, c'est-à-dire la proportion des vrais « c » retrouvés.
- **F1** : `2·precision·recall / (precision + recall)`, la moyenne harmonique des deux.
- **F1 macro** : la moyenne non pondérée des F1 des 5 classes. Elle donne le même poids à chaque classe, alors que l'accuracy est dominée par `Nothing` (41 % du dataset).
- **Matrice de confusion** : `M[i, j]` est le nombre d'exemples de vraie classe `i` prédits en classe `j`. La diagonale contient les bonnes réponses.

---

## 8. Technologies

- **Python 3** : le langage du projet, choisi pour sa lisibilité. Le sujet autorise « tout ce qui fonctionne sur le dump ».
- **NumPy** : bibliothèque de calcul sur des tableaux multidimensionnels. Ce n'est **pas** une bibliothèque de réseaux de neurones : elle ne fournit ni couches, ni gradients, ni optimiseurs, et tout cela est écrit à la main dans `src/mlp/`. Elle délègue les produits matriciels à une implémentation **BLAS** (*Basic Linear Algebra Subprograms*, une interface standard de routines d'algèbre linéaire optimisées et multithreadées). Sur macOS, c'est le framework Accelerate.
- **Matplotlib** : bibliothèque de tracé de graphiques, utilisée uniquement pour les benchmarks ([src/Graphique/](../src/Graphique/)).
- **multiprocessing** (bibliothèque standard Python) : exécute des fonctions dans plusieurs **processus** indépendants. On l'utilise à la place des threads à cause du **GIL** (*Global Interpreter Lock*), un verrou de l'interpréteur CPython qui empêche deux threads d'exécuter du bytecode Python en même temps. Utilisé pour convertir les FEN ([src/clean/dataset.py](../src/clean/dataset.py)) et pour entraîner plusieurs configurations en parallèle ([src/Graphique/benchmark.py](../src/Graphique/benchmark.py)).
- **Memory map (`mmap`)** : un fichier projeté dans l'espace mémoire d'un processus. Le système charge les pages à la demande et les partage entre les processus qui lisent le même fichier. Utilisé pour le cache `.npy` du dataset.
- **Format INI** : format de configuration texte en sections `[nom]` et en lignes `clé = valeur`, lu par le module standard `configparser`. Utilisé pour les `.conf`.
- **Format NPZ** : archive ZIP de tableaux NumPy (`.npy`). Utilisé pour les fichiers `.nn`. Voir [architecture.md](architecture.md).
- **JSON** : format texte de données structurées. Il sert à stocker la configuration et les métadonnées dans le `.nn`, ainsi que les résultats des benchmarks.
