"""Training pipeline for PCB Defect Detection with YOLOv8."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import Config
from src.data_ingestion import DataIngestion
from src.model import PCBDetector
from src.utils import format_metrics, get_logger, print_section_header

logger = get_logger(__name__)


class DatasetError(Exception):
    """Erreur liée au dataset."""
    pass


class TrainingManager:
    """Gère le pipeline complet d'entraînement."""
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        config: Optional[Config] = None
    ):
        """Initialise le gestionnaire d'entraînement.
        
        Args:
            data_path: Chemin vers le dataset (auto-détecté si None)
            config: Configuration personnalisée
        """
        self.data_path = data_path
        self.config = config or Config()
        self.output_path = Config.get_output_path()
        self.data: Optional[DataIngestion] = None
        self.model: Optional[PCBDetector] = None
        self.metrics: Dict[str, float] = {}
        self.training_results: Any = None
        
        self._print_header()
    
    def _print_header(self) -> None:
        """Affiche les informations système avec style."""
        print("\n" + "🔷" * 30)
        print_section_header("🔬 PCB DEFECT DETECTION - YOLOv8 🔬")
        print("🔷" * 30)
        
        # Infos système
        print(f"\n📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️  Environnement: {'☁️ Kaggle' if Config.is_kaggle() else '💻 Local'}")
        print(f"📁 Sortie: {self.output_path}")
        
        # Config modèle
        model_cfg = self.config.model
        print(f"\n⚙️  Configuration:")
        print(f"   • Modèle: {model_cfg.name}")
        print(f"   • Époques: {model_cfg.epochs}")
        print(f"   • Batch size: {model_cfg.batch_size}")
        print(f"   • Learning rate: {model_cfg.learning_rate}")
        print(f"   • Image size: {model_cfg.img_size}x{model_cfg.img_size}")
        print(f"   • Optimizer: {model_cfg.optimizer}")
        print("")
    
    def setup_data(self) -> DataIngestion:
        """Configure le pipeline de données."""
        print_section_header("📊 [1/5] CONFIGURATION DES DONNÉES")
        
        self.data = DataIngestion(data_path=self.data_path)
        
        if not self.data.find_data_structure():
            raise DatasetError(f"Dataset non trouvé à {self.data.data_path}")
        
        self.data.collect_images()
        
        if not self.data.all_images:
            raise DatasetError("Aucune image trouvée dans le dataset")
        
        stats = self.data.get_stats()
        
        print(f"\n📈 Statistiques du dataset:")
        print(f"   • Total images: {stats['total_images']}")
        print(f"   • Avec annotations XML: {stats['with_xml']} ✅")
        print(f"   • Sans annotations (ignorées): {stats['from_folders']} ⚠️")
        
        train_count, val_count = self.data.create_yolo_dataset()
        
        print(f"\n📂 Dataset YOLO créé:")
        print(f"   • Train: {train_count} images")
        print(f"   • Validation: {val_count} images")
        print(f"   • Ratio: {train_count/(train_count+val_count)*100:.1f}% / {val_count/(train_count+val_count)*100:.1f}%")
        
        return self.data
    
    def setup_model(self) -> PCBDetector:
        """Initialise le modèle."""
        print_section_header("🤖 [2/5] CONFIGURATION DU MODÈLE")
        
        self.model = PCBDetector(config=self.config)
        
        print(f"\n✅ Modèle initialisé: {self.config.model.name}")
        print(f"   • Classes: {Config.NUM_CLASSES}")
        print(f"   • Classes: {', '.join(Config.CLASS_NAMES)}")
        
        return self.model
    
    def train(self, epochs: Optional[int] = None) -> Any:
        """Entraîne le modèle."""
        print_section_header("🚀 [3/5] ENTRAÎNEMENT")
        
        if self.data is None:
            raise RuntimeError("Appelez setup_data() d'abord")
        if self.model is None:
            raise RuntimeError("Appelez setup_model() d'abord")
        
        epochs = epochs or self.config.model.epochs
        
        print(f"\n⏱️  Démarrage de l'entraînement pour {epochs} époques...")
        print(f"   (Temps estimé: 15-30 min sur GPU Kaggle)")
        print(f"   💡 Early stopping activé - arrêt auto si convergence")
        print("\n" + "-" * 60)
        
        yaml_path = self.data.get_yaml_path()
        
        self.training_results = self.model.train(
            data_yaml=yaml_path,
            epochs=epochs,
            project=str(self.output_path),
            name="pcb_yolo"
        )
        
        print("-" * 60)
        print("✅ Entraînement terminé!")
        
        return self.training_results
    
    def evaluate(self) -> Dict[str, float]:
        """Évalue le modèle."""
        print_section_header("📏 [4/5] ÉVALUATION")
        
        if self.data is None or self.model is None:
            raise RuntimeError("Appelez setup_data() et setup_model() d'abord")
        
        yaml_path = self.data.get_yaml_path()
        results = self.model.validate(data_yaml=yaml_path)
        
        self.metrics = PCBDetector.extract_metrics(results)
        
        # Affichage stylé des métriques
        print("\n" + "=" * 50)
        print("📊 MÉTRIQUES DE PERFORMANCE")
        print("=" * 50)
        
        map50 = self.metrics.get('mAP50', 0)
        map50_95 = self.metrics.get('mAP50-95', 0)
        precision = self.metrics.get('precision', 0)
        recall = self.metrics.get('recall', 0)
        
        # Indicateurs visuels
        def get_indicator(value: float) -> str:
            if value >= 0.9:
                return "🟢 Excellent"
            elif value >= 0.7:
                return "🟡 Bon"
            elif value >= 0.5:
                return "🟠 Moyen"
            else:
                return "🔴 À améliorer"
        
        print(f"\n   mAP@50:     {map50:.4f}  ({map50*100:.1f}%)  {get_indicator(map50)}")
        print(f"   mAP@50-95:  {map50_95:.4f}  ({map50_95*100:.1f}%)  {get_indicator(map50_95)}")
        print(f"   Precision:  {precision:.4f}  ({precision*100:.1f}%)  {get_indicator(precision)}")
        print(f"   Recall:     {recall:.4f}  ({recall*100:.1f}%)  {get_indicator(recall)}")
        
        # Score global
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"\n   F1-Score:   {f1_score:.4f}  ({f1_score*100:.1f}%)  {get_indicator(f1_score)}")
        
        print("=" * 50)
        
        return self.metrics
    
    def save_model(self) -> Optional[Path]:
        """Sauvegarde le modèle entraîné."""
        print_section_header("💾 [5/5] SAUVEGARDE")
        
        best_model = self.output_path / "pcb_yolo" / "weights" / "best.pt"
        if not best_model.exists():
            print("⚠️  Modèle best.pt non trouvé")
            return None
        
        dst = self.output_path / "pcb_model.pt"
        shutil.copy(best_model, dst)
        print(f"✅ Modèle PyTorch: {dst}")
        
        # Export ONNX
        try:
            from src.model import PCBDetector
            best_detector = PCBDetector(model_path=str(best_model))
            onnx_path = best_detector.export(format="onnx")
            
            onnx_dst = self.output_path / "pcb_model.onnx"
            if onnx_path and onnx_path.exists():
                shutil.copy(onnx_path, onnx_dst)
                print(f"✅ Modèle ONNX: {onnx_dst}")
        except Exception as e:
            print(f"⚠️  Export ONNX échoué: {e}")
        
        return dst
    
    def generate_graphs(self) -> None:
        """Génère et affiche les graphiques d'entraînement."""
        print_section_header("📈 GRAPHIQUES D'ENTRAÎNEMENT")
        
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # Chercher le fichier results.csv
            results_file = self.output_path / "pcb_yolo" / "results.csv"
            
            if not results_file.exists():
                print("⚠️  Fichier results.csv non trouvé")
                return
            
            # Charger les données
            df = pd.read_csv(results_file)
            df.columns = df.columns.str.strip()  # Nettoyer les noms de colonnes
            
            # Créer la figure avec plusieurs sous-graphiques
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('📊 PCB Defect Detection - Résultats d\'entraînement', fontsize=16, fontweight='bold')
            
            # 1. Loss d'entraînement
            ax1 = axes[0, 0]
            if 'train/box_loss' in df.columns:
                ax1.plot(df['epoch'], df['train/box_loss'], 'b-', label='Box Loss', linewidth=2)
                ax1.plot(df['epoch'], df['train/cls_loss'], 'r-', label='Class Loss', linewidth=2)
                ax1.plot(df['epoch'], df['train/dfl_loss'], 'g-', label='DFL Loss', linewidth=2)
            ax1.set_xlabel('Époque')
            ax1.set_ylabel('Loss')
            ax1.set_title('📉 Loss d\'entraînement')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Loss de validation
            ax2 = axes[0, 1]
            if 'val/box_loss' in df.columns:
                ax2.plot(df['epoch'], df['val/box_loss'], 'b--', label='Box Loss', linewidth=2)
                ax2.plot(df['epoch'], df['val/cls_loss'], 'r--', label='Class Loss', linewidth=2)
                ax2.plot(df['epoch'], df['val/dfl_loss'], 'g--', label='DFL Loss', linewidth=2)
            ax2.set_xlabel('Époque')
            ax2.set_ylabel('Loss')
            ax2.set_title('📉 Loss de validation')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. mAP
            ax3 = axes[0, 2]
            if 'metrics/mAP50(B)' in df.columns:
                ax3.plot(df['epoch'], df['metrics/mAP50(B)'], 'b-', label='mAP@50', linewidth=2, marker='o', markersize=3)
                ax3.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'r-', label='mAP@50-95', linewidth=2, marker='s', markersize=3)
            ax3.set_xlabel('Époque')
            ax3.set_ylabel('mAP')
            ax3.set_title('🎯 mAP (Mean Average Precision)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, 1])
            
            # 4. Precision & Recall
            ax4 = axes[1, 0]
            if 'metrics/precision(B)' in df.columns:
                ax4.plot(df['epoch'], df['metrics/precision(B)'], 'g-', label='Precision', linewidth=2)
                ax4.plot(df['epoch'], df['metrics/recall(B)'], 'm-', label='Recall', linewidth=2)
            ax4.set_xlabel('Époque')
            ax4.set_ylabel('Score')
            ax4.set_title('📊 Precision & Recall')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1])
            
            # 5. Learning Rate
            ax5 = axes[1, 1]
            if 'lr/pg0' in df.columns:
                ax5.plot(df['epoch'], df['lr/pg0'], 'c-', label='LR pg0', linewidth=2)
                ax5.plot(df['epoch'], df['lr/pg1'], 'y-', label='LR pg1', linewidth=2)
                ax5.plot(df['epoch'], df['lr/pg2'], 'k-', label='LR pg2', linewidth=2)
            ax5.set_xlabel('Époque')
            ax5.set_ylabel('Learning Rate')
            ax5.set_title('📈 Learning Rate Schedule')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. Résumé final
            ax6 = axes[1, 2]
            ax6.axis('off')
            
            # Métriques finales
            final_metrics = f"""
╔══════════════════════════════════════╗
║     📊 RÉSULTATS FINAUX              ║
╠══════════════════════════════════════╣
║  mAP@50:      {self.metrics.get('mAP50', 0):.4f}  ({self.metrics.get('mAP50', 0)*100:.1f}%)     ║
║  mAP@50-95:   {self.metrics.get('mAP50-95', 0):.4f}  ({self.metrics.get('mAP50-95', 0)*100:.1f}%)     ║
║  Precision:   {self.metrics.get('precision', 0):.4f}  ({self.metrics.get('precision', 0)*100:.1f}%)     ║
║  Recall:      {self.metrics.get('recall', 0):.4f}  ({self.metrics.get('recall', 0)*100:.1f}%)     ║
╚══════════════════════════════════════╝
"""
            ax6.text(0.1, 0.5, final_metrics, fontsize=12, fontfamily='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # Sauvegarder
            graph_path = self.output_path / "training_results.png"
            plt.savefig(graph_path, dpi=150, bbox_inches='tight')
            print(f"✅ Graphiques sauvegardés: {graph_path}")
            
            # Afficher dans Kaggle/Jupyter
            plt.show()
            
        except ImportError as e:
            print(f"⚠️  Matplotlib/Pandas non disponible: {e}")
        except Exception as e:
            print(f"⚠️  Erreur génération graphiques: {e}")
    
    def display_sample_predictions(self) -> None:
        """Affiche des exemples de prédictions."""
        print_section_header("🖼️  EXEMPLES DE PRÉDICTIONS")
        
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            
            # Chercher les images de prédiction générées par YOLO
            pred_dir = self.output_path / "pcb_yolo"
            
            # Images de validation avec prédictions
            val_images = list(pred_dir.glob("val_batch*_pred.jpg"))
            
            if not val_images:
                print("⚠️  Pas d'images de prédiction trouvées")
                return
            
            # Afficher jusqu'à 4 images
            n_images = min(4, len(val_images))
            fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
            
            if n_images == 1:
                axes = [axes]
            
            for i, img_path in enumerate(val_images[:n_images]):
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f'Batch {i+1}')
            
            plt.suptitle('🔍 Exemples de détections sur validation', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Sauvegarder
            sample_path = self.output_path / "sample_predictions.png"
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            print(f"✅ Exemples sauvegardés: {sample_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Erreur affichage exemples: {e}")
    
    def run_pipeline(self, epochs: Optional[int] = None) -> Dict[str, float]:
        """Exécute le pipeline complet d'entraînement."""
        
        self.setup_data()
        self.setup_model()
        self.train(epochs=epochs)
        self.evaluate()
        self.save_model()
        
        # Générer les graphiques
        self.generate_graphs()
        self.display_sample_predictions()
        
        # Résumé final
        print("\n" + "🎉" * 30)
        print_section_header("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("🎉" * 30)
        
        print(f"""
╔══════════════════════════════════════════════════════════╗
║                    📊 RÉSUMÉ FINAL                       ║
╠══════════════════════════════════════════════════════════╣
║  mAP@50:      {self.metrics.get('mAP50', 0):.4f}  ({self.metrics.get('mAP50', 0)*100:.1f}%)                        ║
║  mAP@50-95:   {self.metrics.get('mAP50-95', 0):.4f}  ({self.metrics.get('mAP50-95', 0)*100:.1f}%)                        ║
║  Precision:   {self.metrics.get('precision', 0):.4f}  ({self.metrics.get('precision', 0)*100:.1f}%)                        ║
║  Recall:      {self.metrics.get('recall', 0):.4f}  ({self.metrics.get('recall', 0)*100:.1f}%)                        ║
╠══════════════════════════════════════════════════════════╣
║  📁 Fichiers générés:                                    ║
║     • pcb_model.pt (PyTorch)                             ║
║     • pcb_model.onnx (ONNX)                              ║
║     • training_results.png (Graphiques)                  ║
║     • sample_predictions.png (Exemples)                  ║
╚══════════════════════════════════════════════════════════╝
""")
        
        print(f"📂 Tous les fichiers dans: {self.output_path}")
        
        return self.metrics
