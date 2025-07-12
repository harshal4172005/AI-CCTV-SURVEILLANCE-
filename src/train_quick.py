#!/usr/bin/env python3
"""
Quick Training Script - 5 Epochs
For faster initial results and validation
"""

import os
import sys
from ultralytics import YOLO

def train_quick():
    """Train YOLOv8 model for 5 epochs"""
    print("🚀 Starting Quick Training (5 Epochs)...")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = "datasets/dataset/Construction_site_safety_dataset/ppe_data.yaml"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("💡 Please ensure the dataset is properly set up")
        return False
    
    print(f"✅ Dataset found: {dataset_path}")
    
    # Load YOLOv8 model
    print("📦 Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Load pretrained model
    
    # Training configuration for quick training
    print("⚙️ Configuring quick training parameters...")
    
    # Start training
    print("🎯 Starting training with 5 epochs...")
    print("⏱️ Estimated time: 30-60 minutes (depending on your CPU)")
    print("=" * 50)
    
    try:
        results = model.train(
            data=dataset_path,
            epochs=5,  # Quick training with 5 epochs
            imgsz=640,
            batch=16,
            device='cpu',  # Use CPU
            patience=3,  # Early stopping patience
            save=True,
            save_period=1,  # Save every epoch
            verbose=True,
            project='runs/detect',
            name='quick_train_5_epochs'
        )
        
        print("✅ Quick training completed!")
        print("📊 Training results saved in runs/detect/quick_train_5_epochs/")
        
        # Show final metrics
        if results:
            print("\n📈 Final Training Metrics:")
            print(f"   - Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"   - Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"   - Training Loss: {results.results_dict.get('train/box_loss', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    """Main function"""
    print("🎯 AI CCTV Surveillance - Quick Training (5 Epochs)")
    print("=" * 60)
    print("This will train your model for 5 epochs for quick validation")
    print("You can resume training later for better accuracy")
    print("=" * 60)
    
    # Confirm with user
    response = input("Do you want to start quick training? (y/n): ").lower()
    if response != 'y':
        print("❌ Training cancelled")
        return
    
    # Start training
    success = train_quick()
    
    if success:
        print("\n🎉 Quick training completed successfully!")
        print("💡 Next steps:")
        print("   1. Test the model with inference.py")
        print("   2. Launch the app to see results")
        print("   3. Resume training later for better accuracy")
        print("   4. Copy best.pt to app/models/ for the app")
    else:
        print("\n❌ Quick training failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 