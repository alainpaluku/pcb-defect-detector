"""
Main entry point for PCB Defect Detection System.

This script can be run directly for local training or imported in Kaggle notebooks.

Usage:
    python main.py                    # Run complete pipeline
    python main.py --download         # Download dataset first
    python main.py --epochs 100       # Custom epochs
"""

import argparse
from src.trainer import TrainingManager
from src.kaggle_setup import KaggleSetup
from src.config import Config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PCB Defect Detection System - Automated Optical Inspection'
    )
    
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download dataset from Kaggle before training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=Config.EPOCHS,
        help=f'Number of training epochs (default: {Config.EPOCHS})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=Config.BATCH_SIZE,
        help=f'Batch size for training (default: {Config.BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=Config.LEARNING_RATE,
        help=f'Initial learning rate (default: {Config.LEARNING_RATE})'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Update config with command line arguments
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.learning_rate
    
    # Download dataset if requested
    if args.download:
        print("Downloading dataset from Kaggle...")
        kaggle_setup = KaggleSetup()
        kaggle_setup.download_dataset(Config.KAGGLE_DATASET)
    
    # Check if dataset exists
    data_path = Config.get_data_path()
    if not data_path.exists():
        print(f"\n❌ Dataset not found at: {data_path}")
        print("\nOptions:")
        print("1. Run with --download flag to download from Kaggle")
        print("2. Manually download and extract to 'data/pcb-defects'")
        print("3. Run in Kaggle environment with dataset added")
        return
    
    # Initialize and run training pipeline
    print("\n" + "="*60)
    print("PCB DEFECT DETECTION SYSTEM")
    print("Automated Optical Inspection for Electronics Manufacturing")
    print("="*60)
    print(f"Configuration:")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print(f"  Image Size: {Config.IMG_SIZE}")
    print("="*60 + "\n")
    
    # Run training
    trainer = TrainingManager()
    metrics = trainer.run_pipeline()
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print("Final Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize():15s}: {value:.4f}")
    print("="*60)
    print(f"\nModel artifacts saved to: {Config.get_output_path()}")
    print("Ready for deployment in industrial AOI systems!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
