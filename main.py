"""Main entry point for PCB Defect Detection System."""

import argparse
import sys
from src.config import Config


def main():
    parser = argparse.ArgumentParser(
        description='🔬 PCB Defect Detection - MobileNetV2 Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--download', action='store_true', 
                        help='Download dataset from Kaggle')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--no-fine-tune', action='store_true',
                        help='Skip fine-tuning phase')
    parser.add_argument('--visualize-aug', action='store_true',
                        help='Visualize data augmentation')
    args = parser.parse_args()
    
    # Update config
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.lr
    
    # Download dataset if requested
    if args.download:
        from src.kaggle_setup import KaggleSetup
        setup = KaggleSetup()
        result = setup.download_dataset(Config.KAGGLE_DATASET)
        if not result:
            print("❌ Failed to download dataset")
            sys.exit(1)
        print("✅ Dataset downloaded successfully")
    
    # Check dataset exists
    data_path = Config.get_data_path()
    if not data_path.exists():
        print(f"❌ Error: Dataset not found at {data_path}")
        print("\n📋 Options:")
        print("  1. Run with --download to fetch from Kaggle")
        print("  2. Manually place dataset in data/pcb-defects/")
        print("  3. On Kaggle, add 'akhatova/pcb-defects' dataset")
        sys.exit(1)
    
    # Import trainer (heavy imports)
    from src.trainer import TrainingManager
    
    # Train
    trainer = TrainingManager()
    metrics = trainer.run_pipeline(
        fine_tune=not args.no_fine_tune,
        visualize_augmentation=args.visualize_aug
    )
    
    return metrics


if __name__ == "__main__":
    main()
