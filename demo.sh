#!/bin/bash
# Démo de soutenance : lance toutes les commandes dans l'ordre, sans entraîner le réseau.
# Tout ce qui est créé va dans demo_live/ ; doc/ et my_torch_network.nn ne sont jamais modifiés.
# Usage : ./demo.sh   (Entrée pour passer à l'étape suivante)

cd "$(dirname "$0")" || exit 1
OUT=demo_live

# Affiche un titre d'étape et attend Entrée
step() {
    echo
    printf '\033[1;34m=== %s ===\033[0m\n' "$1"
    read -r -p "(Entrée pour lancer) "
}

# Affiche la commande puis l'exécute
run() {
    printf '\033[1;32m$ %s\033[0m\n' "$1"
    eval "$1"
}

# Vérifications avant de commencer
python3 -c "import numpy, matplotlib" 2>/dev/null || { echo "Il manque numpy ou matplotlib : pip3 install numpy matplotlib"; exit 1; }
if [ ! -f dataset/test.txt ]; then
    echo "dataset/test.txt manque : création du jeu de test (découpage seulement, pas d'entraînement)"
    python3 -m src.clean.build_dataset || exit 1
fi
rm -rf "$OUT"
mkdir -p "$OUT"

step "1. Compiler"
run "make"

step "2. L'aide des deux programmes"
run "./my_torch_analyzer --help"
run "./my_torch_generator --help"

step "3. Générer 2 réseaux vierges à partir d'une config"
run "cat configs/basic_network.conf"
run "(cd $OUT && ../my_torch_generator ../configs/basic_network.conf 2)"
run "ls $OUT"

step "4. Le fichier de sauvegarde du réseau entraîné"
run "unzip -l my_torch_network.nn"

step "5. Prédire 5 positions"
run "head -5 dataset/test.txt > $OUT/demo.txt"
run "cat $OUT/demo.txt"
run "./my_torch_analyzer --predict my_torch_network.nn $OUT/demo.txt"

step "6. Bonne réponse | prédiction, côte à côte"
run "./my_torch_analyzer --predict my_torch_network.nn $OUT/demo.txt > $OUT/demo_pred.txt"
run "cut -d' ' -f7- $OUT/demo.txt | paste -d'|' - $OUT/demo_pred.txt"

step "7. Test complet sur les 72 301 positions jamais vues"
run "python3 -m src.Graphique.evaluate my_torch_network.nn dataset/test.txt $OUT/evaluation"

step "8. Même score avec la commande du sujet"
run "./my_torch_analyzer --predict my_torch_network.nn dataset/test.txt > $OUT/predictions.txt"
run "cut -d' ' -f7- dataset/test.txt > $OUT/attendu.txt"
run "paste -d'|' $OUT/attendu.txt $OUT/predictions.txt | awk -F'|' '\$1==\$2{ok++} END{printf \"%d/%d = %.2f%%\\n\", ok, NR, 100*ok/NR}'"

step "9. Gestion des erreurs (code 84)"
run "./my_torch_analyzer --predict nexistepas.nn $OUT/demo.txt"
echo "code de sortie : $?"
run "./my_torch_generator"
echo "code de sortie : $?"

step "10. La matrice de confusion"
if command -v open >/dev/null; then
    run "open $OUT/evaluation/confusion.png"
elif command -v xdg-open >/dev/null; then
    run "xdg-open $OUT/evaluation/confusion.png"
else
    echo "Image : $OUT/evaluation/confusion.png"
fi

echo
echo "Démo terminée. Pour tout effacer : rm -rf $OUT"
