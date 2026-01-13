"""
PCB Defect Detector - Debug Script for Kaggle
Run this first to diagnose dataset structure issues.
"""

from pathlib import Path

DATA_DIR = Path('/kaggle/input/pcb-defects')

print("=" * 70)
print("KAGGLE DATASET DIAGNOSTIC TOOL")
print("=" * 70)

# Check if directory exists
if not DATA_DIR.exists():
    print(f"\n❌ ERROR: {DATA_DIR} does not exist!")
    print("\n📋 SOLUTION:")
    print("  1. In your Kaggle notebook, click '+ Add Data' (top right)")
    print("  2. Search for: 'akhatova/pcb-defects' or just 'pcb-defects'")
    print("  3. Click on the dataset, then click 'Add'")
    print("  4. Wait for it to mount, then re-run this cell")
    print("=" * 70)
    exit(1)

print(f"\n✓ Dataset directory exists: {DATA_DIR}")

# Explore structure
print("\n📂 DIRECTORY STRUCTURE:")
print("-" * 70)

valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
image_count = 0
dir_count = 0

def explore_dir(path: Path, depth: int = 0, max_depth: int = 3):
    """Recursively explore directory."""
    global image_count, dir_count
    
    if depth > max_depth:
        return
    
    try:
        items = sorted(path.iterdir())
    except PermissionError:
        return
    
    for item in items[:20]:  # Limit items per directory
        indent = "  " * depth
        
        if item.is_dir():
            dir_count += 1
            print(f"{indent}📁 {item.name}/")
            explore_dir(item, depth + 1, max_depth)
        elif item.suffix.lower() in valid_ext:
            image_count += 1
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"{indent}🖼️  {item.name} ({size_mb:.2f} MB)")

explore_dir(DATA_DIR)

print("-" * 70)
print(f"\n📊 SUMMARY:")
print(f"  Total directories found: {dir_count}")
print(f"  Total images found: {image_count}")

# Suggest correct path
print("\n💡 SUGGESTED CONFIGURATION:")

if image_count > 0:
    print("  ✓ Images found! Dataset is correctly loaded.")
    
    # Find the directory with class folders
    for item in DATA_DIR.rglob('*'):
        if item.is_dir():
            subdirs = [d for d in item.iterdir() if d.is_dir()]
            if len(subdirs) >= 3:  # Likely class folders
                has_images = False
                for subdir in subdirs[:3]:
                    imgs = [f for f in subdir.iterdir() if f.suffix.lower() in valid_ext]
                    if imgs:
                        has_images = True
                        break
                
                if has_images:
                    print(f"\n  Recommended DATA_DIR path:")
                    print(f"  DATA_DIR = Path('{item}')")
                    print(f"\n  Class folders found:")
                    for subdir in subdirs:
                        imgs = [f for f in subdir.iterdir() if f.suffix.lower() in valid_ext]
                        if imgs:
                            print(f"    - {subdir.name}: {len(imgs)} images")
                    break
else:
    print("  ❌ No images found. Please check:")
    print("     1. Dataset is correctly added")
    print("     2. Dataset name is 'akhatova/pcb-defects'")

print("\n" + "=" * 70)
print("Copy the recommended DATA_DIR path above into kaggle_script.py")
print("=" * 70)
