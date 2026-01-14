#!/bin/bash
# Script pour pousser sur GitHub
# Usage: bash git_push.sh [message de commit]

set -e

# Se placer dans le répertoire du script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Message de commit (argument ou valeur par défaut)
COMMIT_MSG="${1:-Update: $(date '+%Y-%m-%d %H:%M')}"

# Ajouter tous les fichiers modifiés
git add -A

# Commit avec message
git commit -m "$COMMIT_MSG" || echo "Rien à commiter"

# Configurer le remote (si pas déjà fait)
if ! git remote get-url origin &>/dev/null; then
    git remote add origin https://github.com/alainpaluku/pcb-defect-detector.git
fi

# Pousser sur main
git branch -M main
git push -u origin main

echo ""
echo "✅ Code poussé sur GitHub!"
echo "📍 https://github.com/alainpaluku/pcb-defect-detector"
