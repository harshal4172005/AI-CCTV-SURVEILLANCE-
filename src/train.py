from ultralytics import YOLO
import argparse
import os
import sys

def validate_paths(data_yaml_path: str, model_name: str):
    """Validate that all required files exist."""
    if not os.path.exists(data_yaml_path):
        print(f"❌ Error: Data file '{data_yaml_path}' not found!")
        return False
    
    if not os.path.exists(model_name):
        print(f"❌ Error: Model file '{model_name}' not found!")
        return False
    
    # Check if data directory exists
    import yaml
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Check train, val, test paths
        for split in ['train', 'val', 'test']:
            if split in data_config:
                path = data_config[split]
                if not os.path.exists(path):
                    print(f"❌ Error: {split} path '{path}' not found!")
                    return False
                if not os.path.isdir(path):
                    print(f"❌ Error: {split} path '{path}' is not a directory!")
                    return False
                
                # Check if directory has images
                image_files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if not image_files:
                    print(f"⚠️ Warning: No images found in {split} directory: {path}")
        
        print("✅ All data paths validated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error reading data config: {e}")
        return False

def train_model(
    data_yaml_path: str = "dataset/data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 20,
    imgsz: int = 640,
    project: str = "runs/train",
    name: str = "helmet_yolov8",
    batch_size: int = 16
):
    """
    Train YOLOv8 on the specified dataset.
    """
    print(f"🚀 Starting training with:")
    print(f"   Data: {data_yaml_path}")
    print(f"   Model: {model_name}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch_size}")
    print(f"   Project: {project}")
    print(f"   Name: {name}")
    print("-" * 50)
    
    try:
        model = YOLO(model_name)
        print("✅ Model loaded successfully")
        
        # Start training
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=project,
            name=name,
            patience=50,  # Early stopping patience
            save=True,
            save_period=5,  # Save every 5 epochs
            plots=True,  # Generate training plots
            verbose=True
        )
        
        print(f"✅ Training complete!")
        print(f"📁 Model saved in {project}/{name}/weights/")
        print(f"📊 Training results saved in {project}/{name}/")
        
        # Print final metrics
        if results:
            print(f"\n📈 Training completed successfully!")
            print(f"📁 Model saved in {project}/{name}/weights/")
            print(f"📊 Training results saved in {project}/{name}/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for PPE detection")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project directory")
    parser.add_argument("--name", type=str, default="train", help="Experiment name")
    parser.add_argument("--validate", action="store_true", help="Validate data paths before training")
    
    args = parser.parse_args()
    
    print("🎯 AI CCTV Surveillance - Model Training")
    print("=" * 50)
    
    # Validate paths if requested
    if args.validate:
        if not validate_paths(args.data, args.model):
            sys.exit(1)
    
    # Start training
    success = train_model(
        data_yaml_path=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        batch_size=args.batch
    )
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("💡 Next steps:")
        print("   1. Test the model: python test_model.py")
        print("   2. Launch web app: python launch_app.py")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)
