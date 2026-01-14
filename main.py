"""Main entry point for PCB Defect Detection System."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='PCB Defect Detection with YOLOv8',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data', type=str, help='Path to dataset')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    train_parser.add_argument('--img-size', type=int, default=640, help='Image size')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Run detection on images')
    detect_parser.add_argument('source', type=str, help='Image or directory path')
    detect_parser.add_argument('--model', type=str, help='Path to trained model')
    detect_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    detect_parser.add_argument('--save', action='store_true', help='Save results')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model')
    export_parser.add_argument('--model', type=str, required=True, help='Path to model')
    export_parser.add_argument('--format', type=str, default='onnx', help='Export format')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from src.trainer import TrainingManager
        from src.config import Config
        
        if args.epochs:
            Config.EPOCHS = args.epochs
        if args.batch:
            Config.BATCH_SIZE = args.batch
        if args.img_size:
            Config.IMG_SIZE = args.img_size
        
        trainer = TrainingManager(data_path=args.data)
        metrics = trainer.run_pipeline()
        
        print(f"\nFinal mAP@50: {metrics.get('mAP50', 0):.4f}")
        
    elif args.command == 'detect':
        from src.detector import PCBInspector
        from src.utils import print_detection_results
        
        inspector = PCBInspector(model_path=args.model)
        
        source = Path(args.source)
        if source.is_dir():
            results = inspector.inspect_batch(source, conf=args.conf, save=args.save)
            for img_name, detections in results.items():
                print(f"\n{img_name}:")
                print_detection_results(detections)
        else:
            detections = inspector.inspect(source, conf=args.conf, save=args.save)
            print_detection_results(detections)
            
    elif args.command == 'export':
        from src.model import PCBDetector
        
        detector = PCBDetector(model_path=args.model)
        detector.export(format=args.format)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
