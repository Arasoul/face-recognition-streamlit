#!/usr/bin/env python3
"""
Setup script for Face Recognition System

This script helps users set up the environment and prepare data for training.
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logging.error("❌ Python 3.8+ is required")
        return False
    logging.info(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Install required packages"""
    logging.info("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logging.info("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed to install requirements: {e}")
        return False
    except FileNotFoundError:
        logging.error("❌ requirements.txt not found")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            logging.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            logging.warning("⚠️ CUDA not available, using CPU")
            return False
    except ImportError:
        logging.warning("⚠️ PyTorch not installed yet")
        return False

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        "images dataset",
        "models",
        "logs",
        "outputs"
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        logging.info(f"📁 Created directory: {dir_name}")
    
    # Create example person directories
    example_persons = ["person1", "person2", "person3"]
    for person in example_persons:
        person_dir = os.path.join("images dataset", person)
        os.makedirs(person_dir, exist_ok=True)
        
        # Create README in each person directory
        readme_path = os.path.join(person_dir, "README.txt")
        if not os.path.exists(readme_path):
            with open(readme_path, 'w') as f:
                f.write(f"""
Place images of {person} in this directory.

Guidelines:
- Use clear, front-facing photos
- Good lighting conditions
- Minimum 5-10 images per person
- Supported formats: .jpg, .jpeg, .png, .bmp
- Avoid blurry or low-quality images

Example filenames:
- 0001.jpg
- 0002.jpg
- 0003.jpg
""")

def create_demo_script():
    """Create a demo script for testing"""
    demo_content = '''#!/usr/bin/env python3
"""
Demo script for Face Recognition System
"""

import os
import sys
import logging
from train_model import FaceRecognitionTrainer

def quick_demo():
    """Quick demonstration of the training process"""
    logging.basicConfig(level=logging.INFO)
    
    data_dir = "images dataset"
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print("❌ Dataset directory not found. Please run setup.py first.")
        return
    
    # Check if there are any person directories with images
    has_data = False
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if os.path.isdir(person_dir):
            images = [f for f in os.listdir(person_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if images:
                has_data = True
                break
    
    if not has_data:
        print("❌ No images found in dataset. Please add images to person directories.")
        print("📁 Example: images dataset/john_doe/0001.jpg")
        return
    
    print("🚀 Starting demo training...")
    
    try:
        trainer = FaceRecognitionTrainer()
        image_paths, labels = trainer.load_dataset(data_dir, min_images_per_person=2)
        
        if not image_paths:
            print("❌ Not enough images for training")
            return
        
        print(f"📊 Found {len(image_paths)} images for {len(set(labels))} persons")
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    quick_demo()
'''
    
    with open("demo.py", 'w') as f:
        f.write(demo_content)
    
    logging.info("📝 Created demo.py")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Setup Face Recognition System')
    parser.add_argument('--skip-install', action='store_true',
                       help='Skip package installation')
    parser.add_argument('--create-demo', action='store_true',
                       help='Create demo script')
    
    args = parser.parse_args()
    
    logging.info("🚀 Setting up Face Recognition System...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not args.skip_install:
        if not install_requirements():
            return False
    
    # Check CUDA
    check_cuda()
    
    # Create directories
    create_directory_structure()
    
    # Create demo script
    if args.create_demo:
        create_demo_script()
    
    logging.info("\n✅ Setup completed successfully!")
    logging.info("\n📋 Next steps:")
    logging.info("1. Add images to 'images dataset/<person_name>/' directories")
    logging.info("2. Run: python train_model.py")
    logging.info("3. Use trained model with: python final9.py or streamlit run streamlit_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
