# 🖥️ PCB Inspector - Desktop App

Application desktop pour la détection de défauts PCB avec TensorFlow.js et Tauri.

## 📋 Prérequis

- Node.js 18+
- Rust (pour Tauri)
- npm ou yarn

## 🚀 Installation

```bash
# Installer les dépendances
npm install

# Lancer en mode développement
npm run dev

# Dans un autre terminal, lancer Tauri
npm run tauri dev
```

## 📦 Build

```bash
# Build pour production
npm run tauri build
```

L'exécutable sera dans `src-tauri/target/release/`.

## 🧠 Conversion du Modèle

Pour utiliser le modèle entraîné, convertissez-le en TensorFlow.js :

```bash
# Installer tensorflowjs
pip install tensorflowjs

# Convertir le modèle Keras
tensorflowjs_converter --input_format=keras \
    ../output/pcb_model.keras \
    public/model
```

Placez les fichiers générés dans `public/model/`.

## 🎨 Stack Technique

- **Frontend**: React + TypeScript + Tailwind CSS
- **ML**: TensorFlow.js
- **Desktop**: Tauri (Rust)
- **Build**: Vite

## 📁 Structure

```
desktop-app/
├── src/
│   ├── App.tsx          # Composant principal
│   ├── model.ts         # Logique TensorFlow.js
│   └── index.css        # Styles Tailwind
├── public/
│   └── model/           # Modèle TF.js converti
├── src-tauri/           # Backend Rust
└── package.json
```
