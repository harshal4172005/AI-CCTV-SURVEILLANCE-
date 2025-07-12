import os

def verify_dataset_structure(dataset_dir: str):
    """
    Check if dataset has YOLOv8 structure:
    dataset_dir/
        ├── train/
        │   ├── images/
        │   └── labels/
        └── valid/
            ├── images/
            └── labels/
    """
    expected_dirs = [
        "train/images",
        "train/labels",
        "valid/images",
        "valid/labels"
    ]

    all_exist = True
    for path in expected_dirs:
        full_path = os.path.join(dataset_dir, path)
        if not os.path.isdir(full_path):
            print(f"❌ Missing: {full_path}")
            all_exist = False
        else:
            print(f"✅ Found: {full_path}")

    return all_exist
