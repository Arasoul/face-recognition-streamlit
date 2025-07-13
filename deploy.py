#!/usr/bin/env python3
"""
Deployment script for Face Recognition System

This script helps deploy the system in different environments:
- Local development
- Streamlit Cloud
- Docker deployment
- Production server
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

def check_environment():
    """Check if running in cloud environment"""
    cloud_indicators = [
        'STREAMLIT_SHARING',
        'HEROKU',
        'RAILWAY_ENVIRONMENT',
        'GITHUB_ACTIONS'
    ]
    
    for indicator in cloud_indicators:
        if os.getenv(indicator):
            return 'cloud'
    
    return 'local'

def install_dependencies(cloud=False):
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        # Basic installation
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        # Install additional packages for local development
        if not cloud:
            additional_packages = [
                "jupyter",
                "notebook",
                "black",
                "flake8"
            ]
            
            for package in additional_packages:
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package
                    ])
                except subprocess.CalledProcessError:
                    print(f"⚠️ Optional package {package} failed to install")
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_model_files():
    """Set up model files for deployment"""
    required_files = [
        'svm_model.pkl',
        'label_encoder.pkl', 
        'embeddings.npy',
        'labels.npy'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️ Missing model files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n📝 To create model files:")
        print("1. Add training images to 'images dataset/' directory")
        print("2. Run: python train_model.py")
        print("3. Model files will be generated automatically")
        return False
    else:
        print("✅ All model files present")
        return True

def create_startup_script():
    """Create startup script for deployment"""
    startup_content = '''#!/bin/bash
# Startup script for Face Recognition System

echo "🚀 Starting Face Recognition System..."

# Check Python version
python3 --version

# Install dependencies if needed
if [ ! -f ".dependencies_installed" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    touch .dependencies_installed
fi

# Start Streamlit app
echo "🌐 Starting Streamlit application..."
streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
'''
    
    with open('start.sh', 'w') as f:
        f.write(startup_content)
    
    # Make executable
    try:
        os.chmod('start.sh', 0o755)
    except:
        pass  # Windows doesn't support chmod
    
    print("✅ Created startup script: start.sh")

def create_docker_files():
    """Create Docker configuration files"""
    
    # Dockerfile
    dockerfile_content = '''FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libfontconfig1 \\
    libxrender1 \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Start command
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
'''
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    # Docker Compose
    compose_content = '''version: '3.8'

services:
  face-recognition:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./images dataset:/app/images dataset
      - ./models:/app/models
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_PORT=8501
    restart: unless-stopped
    
  # Optional: Add Redis for caching
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
'''
    
    with open('docker-compose.yml', 'w') as f:
        f.write(compose_content)
    
    print("✅ Created Docker files: Dockerfile, docker-compose.yml")

def create_heroku_files():
    """Create Heroku deployment files"""
    
    # Procfile
    with open('Procfile', 'w') as f:
        f.write('web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0\n')
    
    # Heroku requirements (simplified)
    heroku_reqs = '''streamlit>=1.28.0
torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu
facenet-pytorch>=2.5.2
opencv-python-headless>=4.8.0
scikit-learn>=1.3.0
Pillow>=10.0.0
numpy>=1.24.0
joblib>=1.3.0
plotly>=5.17.0
pandas>=2.0.0
'''
    
    with open('requirements-heroku.txt', 'w') as f:
        f.write(heroku_reqs)
    
    print("✅ Created Heroku files: Procfile, requirements-heroku.txt")

def create_github_actions():
    """Create GitHub Actions workflow"""
    os.makedirs('.github/workflows', exist_ok=True)
    
    workflow_content = '''name: Test and Deploy

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run tests
      run: |
        python -m pytest test_system.py -v
    
    - name: Test Streamlit app
      run: |
        streamlit run streamlit_app.py --headless &
        sleep 10
        curl -f http://localhost:8501 || exit 1

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to Streamlit Cloud
      run: |
        echo "Deployment would happen here"
        # Add actual deployment steps
'''
    
    with open('.github/workflows/ci-cd.yml', 'w') as f:
        f.write(workflow_content)
    
    print("✅ Created GitHub Actions workflow")

def main():
    parser = argparse.ArgumentParser(description='Deploy Face Recognition System')
    parser.add_argument('--target', choices=['local', 'cloud', 'docker', 'heroku'], 
                       default='local', help='Deployment target')
    parser.add_argument('--skip-deps', action='store_true', 
                       help='Skip dependency installation')
    parser.add_argument('--create-files', action='store_true',
                       help='Create deployment files')
    
    args = parser.parse_args()
    
    print("🚀 Face Recognition System Deployment")
    print("=" * 50)
    
    env = check_environment()
    print(f"🔍 Environment detected: {env}")
    print(f"🎯 Target deployment: {args.target}")
    
    # Install dependencies
    if not args.skip_deps:
        cloud_mode = (env == 'cloud' or args.target == 'cloud')
        if not install_dependencies(cloud_mode):
            return False
    
    # Check model files
    if not setup_model_files():
        print("\n⚠️ Warning: Model files missing. System will work but recognition will be limited.")
    
    # Create deployment files
    if args.create_files:
        print("\n📝 Creating deployment files...")
        
        create_startup_script()
        
        if args.target == 'docker':
            create_docker_files()
        elif args.target == 'heroku':
            create_heroku_files()
        
        create_github_actions()
    
    print("\n✅ Deployment setup completed!")
    
    # Provide next steps
    print("\n📋 Next steps:")
    if args.target == 'local':
        print("1. Run: streamlit run streamlit_app.py")
        print("2. Or run: python final9.py (for desktop app)")
    elif args.target == 'docker':
        print("1. Run: docker-compose up --build")
        print("2. Access: http://localhost:8501")
    elif args.target == 'heroku':
        print("1. Install Heroku CLI")
        print("2. Run: heroku create your-app-name")
        print("3. Run: git push heroku main")
    elif args.target == 'cloud':
        print("1. Push to GitHub")
        print("2. Connect to Streamlit Cloud")
        print("3. Deploy from repository")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
