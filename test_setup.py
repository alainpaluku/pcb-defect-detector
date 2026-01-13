"""
Setup Validation Script for PCB Defect Detection System.

Run this script to verify that all dependencies and project files
are correctly installed and configured.

Usage:
    python test_setup.py
"""

import sys
from pathlib import Path


def print_header(text):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_check(passed, message):
    """Print check result with appropriate symbol."""
    symbol = "✓" if passed else "✗"
    print(f"{symbol} {message}")


def check_python_version():
    """Check Python version compatibility."""
    print_header("Python Version Check")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 8:
        print_check(True, f"Python {version_str} (Compatible)")
        return True
    else:
        print_check(False, f"Python {version_str} (Requires 3.8+)")
        return False


def check_dependencies():
    """Check if all required packages are installed."""
    print_header("Dependency Check")
    
    dependencies = {
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'PIL': 'Pillow',
        'cv2': 'OpenCV'
    }
    
    all_installed = True
    
    for module, name in dependencies.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print_check(True, f"{name:20s} (v{version})")
        except ImportError:
            print_check(False, f"{name:20s} (Not installed)")
            all_installed = False
    
    return all_installed


def check_tensorflow_gpu():
    """Check TensorFlow GPU availability."""
    print_header("GPU Check")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print_check(True, f"GPU Available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"  Device {i}: {gpu.name}")
            return True
        else:
            print_check(False, "No GPU detected (CPU training will be slower)")
            print("  Note: GPU is optional but recommended for faster training")
            return True  # Not a failure, just a warning
            
    except Exception as e:
        print_check(False, f"Error checking GPU: {e}")
        return True


def check_project_structure():
    """Verify project directory structure."""
    print_header("Project Structure Check")
    
    required_files = [
        'README.md',
        'requirements.txt',
        'main.py',
        'setup.py',
        'src/__init__.py',
        'src/config.py',
        'src/kaggle_setup.py',
        'src/data_ingestion.py',
        'src/model.py',
        'src/trainer.py',
        'notebooks/pcb_defect_detection.ipynb'
    ]
    
    all_present = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print_check(True, file_path)
        else:
            print_check(False, f"{file_path} (Missing)")
            all_present = False
    
    return all_present


def check_imports():
    """Test importing project modules."""
    print_header("Module Import Check")
    
    modules = [
        'src.config',
        'src.kaggle_setup',
        'src.data_ingestion',
        'src.model',
        'src.trainer'
    ]
    
    all_imported = True
    
    for module in modules:
        try:
            __import__(module)
            print_check(True, module)
        except ImportError as e:
            print_check(False, f"{module} ({str(e)})")
            all_imported = False
    
    return all_imported


def check_dataset():
    """Check if dataset is available."""
    print_header("Dataset Check")
    
    try:
        from src.config import Config
        
        data_path = Config.get_data_path()
        
        if data_path.exists():
            # Count classes
            class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
            
            if class_dirs:
                print_check(True, f"Dataset found at: {data_path}")
                print(f"  Classes detected: {len(class_dirs)}")
                
                # Count images per class
                total_images = 0
                for class_dir in class_dirs:
                    images = list(class_dir.glob("*.jpg")) + \
                            list(class_dir.glob("*.png")) + \
                            list(class_dir.glob("*.jpeg"))
                    count = len(images)
                    total_images += count
                    print(f"    {class_dir.name:20s}: {count:4d} images")
                
                print(f"  Total images: {total_images}")
                return True
            else:
                print_check(False, f"Dataset directory empty: {data_path}")
                return False
        else:
            print_check(False, f"Dataset not found at: {data_path}")
            print("\n  To download dataset:")
            print("  1. Kaggle: Add 'akhatova/pcb-defects' to your notebook")
            print("  2. Local: Run download script or manual download")
            print("     python -c \"from src.kaggle_setup import KaggleSetup; KaggleSetup().download_dataset('akhatova/pcb-defects')\"")
            return False
            
    except Exception as e:
        print_check(False, f"Error checking dataset: {e}")
        return False


def check_output_directory():
    """Check if output directory can be created."""
    print_header("Output Directory Check")
    
    try:
        from src.config import Config
        
        output_path = Config.get_output_path()
        
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            print_check(True, f"Output directory created: {output_path}")
        else:
            print_check(True, f"Output directory exists: {output_path}")
        
        # Test write permissions
        test_file = output_path / '.test_write'
        try:
            test_file.write_text('test')
            test_file.unlink()
            print_check(True, "Write permissions verified")
            return True
        except Exception as e:
            print_check(False, f"No write permissions: {e}")
            return False
            
    except Exception as e:
        print_check(False, f"Error checking output directory: {e}")
        return False


def run_quick_test():
    """Run a quick functionality test."""
    print_header("Quick Functionality Test")
    
    try:
        from src.config import Config
        from src.model import PCBClassifier
        
        # Test model creation
        print("Testing model creation...")
        model_wrapper = PCBClassifier(num_classes=6)
        model_wrapper.build_model()
        model_wrapper.compile_model()
        
        print_check(True, "Model creation successful")
        
        # Test model summary
        total_params = model_wrapper.model.count_params()
        print(f"  Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print_check(False, f"Functionality test failed: {e}")
        return False


def print_summary(results):
    """Print final summary of all checks."""
    print_header("SUMMARY")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal Checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    if all(results.values()):
        print("\n✓ All checks passed! System is ready for training.")
        print("\nNext steps:")
        print("  1. Ensure dataset is available (see Dataset Check above)")
        print("  2. Run: python main.py")
        print("  3. Or use Kaggle notebook: notebooks/pcb_defect_detection.ipynb")
        return True
    else:
        print("\n✗ Some checks failed. Please resolve issues above.")
        print("\nCommon solutions:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Download dataset: See KAGGLE_SETUP.md")
        print("  - Check Python version: Requires Python 3.8+")
        return False


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("PCB DEFECT DETECTION SYSTEM - SETUP VALIDATION")
    print("=" * 60)
    
    results = {}
    
    # Run all checks
    results['Python Version'] = check_python_version()
    results['Dependencies'] = check_dependencies()
    results['GPU'] = check_tensorflow_gpu()
    results['Project Structure'] = check_project_structure()
    results['Module Imports'] = check_imports()
    results['Dataset'] = check_dataset()
    results['Output Directory'] = check_output_directory()
    results['Functionality'] = run_quick_test()
    
    # Print summary
    success = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
