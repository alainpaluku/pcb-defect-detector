#!/bin/bash
# Script pour pousser le projet sur GitHub

echo "🚀 PCB Defect Detector - GitHub Push Script"
echo "============================================"
echo ""

# Vérifier si on est dans le bon répertoire
if [ ! -f "README.md" ]; then
    echo "❌ Erreur: Exécute ce script depuis le dossier pcb-defect-detector"
    exit 1
fi

# Demander le nom d'utilisateur GitHub
read -p "📝 Nom d'utilisateur GitHub (ex: alainpaluku): " GITHUB_USER

if [ -z "$GITHUB_USER" ]; then
    echo "❌ Nom d'utilisateur requis"
    exit 1
fi

REPO_NAME="pcb-defect-detector"
REPO_URL="https://github.com/$GITHUB_USER/$REPO_NAME.git"

echo ""
echo "📦 Configuration:"
echo "   Utilisateur: $GITHUB_USER"
echo "   Repository: $REPO_NAME"
echo "   URL: $REPO_URL"
echo ""

# Vérifier si le remote existe déjà
if git remote | grep -q "origin"; then
    echo "⚠️  Remote 'origin' existe déjà"
    read -p "   Remplacer? (y/n): " REPLACE
    if [ "$REPLACE" = "y" ]; then
        git remote remove origin
        echo "   ✅ Remote supprimé"
    else
        echo "   ❌ Annulé"
        exit 1
    fi
fi

# Ajouter le remote
echo "🔗 Ajout du remote..."
git remote add origin "$REPO_URL"

# Vérifier la branche
CURRENT_BRANCH=$(git branch --show-current)
if [ -z "$CURRENT_BRANCH" ]; then
    echo "📌 Création de la branche main..."
    git checkout -b main
fi

# Afficher le statut
echo ""
echo "📊 Statut Git:"
git status --short

echo ""
echo "📝 Derniers commits:"
git log --oneline -3

echo ""
echo "============================================"
echo "⚠️  IMPORTANT: Avant de continuer"
echo "============================================"
echo ""
echo "1. Va sur https://github.com/new"
echo "2. Crée un repo nommé: $REPO_NAME"
echo "3. NE COCHE PAS 'Add README' ou '.gitignore'"
echo "4. Clique sur 'Create repository'"
echo ""
read -p "✅ Repo créé sur GitHub? (y/n): " REPO_CREATED

if [ "$REPO_CREATED" != "y" ]; then
    echo "❌ Crée d'abord le repo sur GitHub"
    exit 1
fi

# Pousser sur GitHub
echo ""
echo "🚀 Push vers GitHub..."
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✅ SUCCÈS!"
    echo "============================================"
    echo ""
    echo "🎉 Ton code est maintenant sur GitHub!"
    echo "📍 URL: https://github.com/$GITHUB_USER/$REPO_NAME"
    echo ""
    echo "👉 Prochaines étapes:"
    echo "   1. Visite ton repo: https://github.com/$GITHUB_USER/$REPO_NAME"
    echo "   2. Ajoute une description"
    echo "   3. Ajoute des topics: machine-learning, deep-learning, pcb, defect-detection"
    echo "   4. Partage ton projet!"
    echo ""
else
    echo ""
    echo "❌ Erreur lors du push"
    echo ""
    echo "💡 Solutions possibles:"
    echo "   1. Vérifie que le repo existe sur GitHub"
    echo "   2. Vérifie tes identifiants Git"
    echo "   3. Essaie avec SSH: git remote set-url origin git@github.com:$GITHUB_USER/$REPO_NAME.git"
    echo ""
fi
