#!/usr/bin/env python3
"""
Resume Training Script
Continue training from the best checkpoint for additional epochs
"""

import os
import sys
from ultralytics import YOLO

def find_best_checkpoint():
    """Find the best checkpoint from previous training runs"""
    runs_dir = "runs/detect"
    if not os.path.exists(runs_dir):
        print("❌ No previous training runs found")
        return None
    
    # Look for training directories
    training_dirs = [d for d in os.listdir(runs_dir) if d.startswith("train") or d.startswith("quick_train")]
    if not training_dirs:
        print("❌ No training directories found")
        return None
    
    # Sort by creation time (newest first)
    training_dirs.sort(key=lambda x: os.path.getctime(os.path.join(runs_dir, x)), reverse=True)
    
    # Check each directory for best.pt
    for train_dir in training_dirs:
        best_pt_path = os.path.join(runs_dir, train_dir, "weights", "best.pt")
        if os.path.exists(best_pt_path):
            print(f"✅ Found best checkpoint: {best_pt_path}")
            return best_pt_path
    
    print("❌ No best.pt checkpoint found in training directories")
    return None

def resume_training(additional_epochs=5):
    """Resume training from the best checkpoint"""
    print(f"🔄 Resuming Training for {additional_epochs} additional epochs...")
    print("=" * 60)
    
    # Find the best checkpoint
    checkpoint_path = find_best_checkpoint()
    if not checkpoint_path:
        print("❌ No checkpoint found to resume from")
        print("💡 Please run quick training first")
        return False
    
    # Check if dataset exists
    dataset_path = "datasets/dataset/Construction_site_safety_dataset/ppe_data.yaml"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return False
    
    print(f"✅ Dataset found: {dataset_path}")
    
    # Load the model from checkpoint
    print(f"📦 Loading model from checkpoint: {checkpoint_path}")
    model = YOLO(checkpoint_path)
    
    # Resume training
    print(f"🎯 Resuming training for {additional_epochs} additional epochs...")
    print("⏱️ Estimated time: 30-60 minutes per 5 epochs (depending on your CPU)")
    print("=" * 60)
    
    try:
        results = model.train(
            data=dataset_path,
            epochs=additional_epochs,
            imgsz=640,
            batch=16,
            device='cpu',
            patience=5,  # Slightly higher patience for resumed training
            save=True,
            save_period=1,
            verbose=True,
            project='runs/detect',
            name='resumed_training'
        )
        
        print("✅ Resume training completed!")
        print("📊 Training results saved in runs/detect/resumed_training/")
        
        # Show final metrics
        if results:
            print("\n📈 Final Training Metrics:")
            print(f"   - Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"   - Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"   - Training Loss: {results.results_dict.get('train/box_loss', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Resume training failed: {e}")
        return False

def main():
    """Main function"""
    print("🔄 AI CCTV Surveillance - Resume Training")
    print("=" * 60)
    print("This will resume training from the best checkpoint")
    print("for additional epochs to improve accuracy")
    print("=" * 60)
    
    # Get number of additional epochs
    try:
        epochs = int(input("Enter number of additional epochs (default 5): ") or "5")
    except ValueError:
        epochs = 5
    
    print(f"🎯 Will train for {epochs} additional epochs")
    
    # Confirm with user
    response = input("Do you want to resume training? (y/n): ").lower()
    if response != 'y':
        print("❌ Training cancelled")
        return
    
    # Resume training
    success = resume_training(epochs)
    
    if success:
        print("\n🎉 Resume training completed successfully!")
        print("💡 Next steps:")
        print("   1. Test the improved model")
        print("   2. Launch the app to see better results")
        print("   3. Copy the new best.pt to app/models/")
        print("   4. Continue training if needed")
    else:
        print("\n❌ Resume training failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 