#!/bin/bash
# Script pour pousser sur GitHub
# Usage: bash git_push.sh

cd ~/pcb-defect-detector/pcb-defect-detector

# Ajouter tous les fichiers modifiés
git add -A

# Commit avec message
git commit -m "Fix: Add critical validation for empty datasets"

# Configurer le remote (si pas déjà fait)
git remote remove origin 2>/dev/null
git remote add origin https://github.com/alainpaluku/pcb-defect-detector.git

# Pousser sur main
git branch -M main
git push -u origin main

echo ""
echo "✅ Code poussé sur GitHub!"
echo "📍 https://github.com/alainpaluku/pcb-defect-detector"
