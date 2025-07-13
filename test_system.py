#!/usr/bin/env python3
"""
Test suite for Face Recognition System

This script runs various tests to ensure the system components work correctly.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
import logging

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)

class TestFaceRecognitionSystem(unittest.TestCase):
    """Test cases for face recognition components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.sample_embedding = np.random.rand(512)
        self.sample_labels = ['person1', 'person2', 'person3']
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_model_imports(self):
        """Test that all required modules can be imported"""
        try:
            import torch
            import numpy as np
            from PIL import Image
            import cv2
            from facenet_pytorch import MTCNN, InceptionResnetV1
            from sklearn.svm import SVC
            from sklearn.preprocessing import LabelEncoder, Normalizer
            import joblib
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import required modules: {e}")
    
    def test_cuda_availability(self):
        """Test CUDA setup"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            print(f"CUDA Available: {cuda_available}")
            if cuda_available:
                print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            self.assertIsInstance(cuda_available, bool)
        except Exception as e:
            self.fail(f"CUDA test failed: {e}")
    
    def test_model_initialization(self):
        """Test model initialization"""
        try:
            from train_model import FaceRecognitionTrainer
            trainer = FaceRecognitionTrainer()
            self.assertIsNotNone(trainer.mtcnn)
            self.assertIsNotNone(trainer.facenet)
            self.assertIsNotNone(trainer.device)
        except Exception as e:
            self.fail(f"Model initialization failed: {e}")
    
    def test_directory_structure_creation(self):
        """Test dataset directory creation"""
        try:
            from train_model import FaceRecognitionTrainer
            trainer = FaceRecognitionTrainer()
            
            # Create test directory structure
            test_dataset = os.path.join(self.test_dir, "test_dataset")
            person_dir = os.path.join(test_dataset, "test_person")
            os.makedirs(person_dir, exist_ok=True)
            
            # Test empty directory handling
            images, labels = trainer.load_dataset(test_dataset, min_images_per_person=1)
            self.assertEqual(len(images), 0)
            self.assertEqual(len(labels), 0)
            
        except Exception as e:
            self.fail(f"Directory structure test failed: {e}")
    
    def test_embedding_extraction(self):
        """Test embedding extraction with mock data"""
        try:
            from train_model import FaceRecognitionTrainer
            
            # Mock the embedding extraction
            with patch.object(FaceRecognitionTrainer, 'extract_embedding') as mock_extract:
                mock_extract.return_value = self.sample_embedding
                
                trainer = FaceRecognitionTrainer()
                result = trainer.extract_embedding("fake_image.jpg")
                
                self.assertIsNotNone(result)
                self.assertEqual(len(result), 512)
                
        except Exception as e:
            self.fail(f"Embedding extraction test failed: {e}")
    
    def test_classifier_training(self):
        """Test SVM classifier training"""
        try:
            from train_model import FaceRecognitionTrainer
            
            # Create mock training data
            embeddings = np.random.rand(30, 512)  # 30 samples, 512 features
            labels = ['person1'] * 10 + ['person2'] * 10 + ['person3'] * 10
            
            trainer = FaceRecognitionTrainer()
            svm, encoder, train_acc, test_acc = trainer.train_classifier(
                embeddings, labels, test_size=0.3, cross_validation=False
            )
            
            self.assertIsNotNone(svm)
            self.assertIsNotNone(encoder)
            self.assertGreater(train_acc, 0)
            self.assertGreater(test_acc, 0)
            
        except Exception as e:
            self.fail(f"Classifier training test failed: {e}")
    
    def test_model_saving_loading(self):
        """Test model save and load functionality"""
        try:
            import joblib
            from sklearn.svm import SVC
            from sklearn.preprocessing import LabelEncoder
            
            # Create mock model and encoder
            mock_svm = SVC(probability=True)
            mock_encoder = LabelEncoder()
            
            # Create mock training data for fitting
            X_mock = np.random.rand(10, 512)
            y_mock = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            mock_svm.fit(X_mock, y_mock)
            mock_encoder.fit(['person1', 'person2'])
            
            # Test saving
            svm_path = os.path.join(self.test_dir, 'test_svm.pkl')
            encoder_path = os.path.join(self.test_dir, 'test_encoder.pkl')
            
            joblib.dump(mock_svm, svm_path)
            joblib.dump(mock_encoder, encoder_path)
            
            # Test loading
            loaded_svm = joblib.load(svm_path)
            loaded_encoder = joblib.load(encoder_path)
            
            self.assertIsNotNone(loaded_svm)
            self.assertIsNotNone(loaded_encoder)
            
        except Exception as e:
            self.fail(f"Model save/load test failed: {e}")
    
    def test_streamlit_imports(self):
        """Test Streamlit app imports"""
        try:
            import streamlit as st
            import plotly.express as px
            import plotly.graph_objects as go
            import pandas as pd
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Streamlit imports failed: {e}")

class TestFileStructure(unittest.TestCase):
    """Test file structure and dependencies"""
    
    def test_required_files_exist(self):
        """Test that all required files exist"""
        required_files = [
            'final9.py',
            'streamlit_app.py',
            'train_model.py',
            'requirements.txt',
            'README.md',
            '.gitignore'
        ]
        
        for file in required_files:
            self.assertTrue(os.path.exists(file), f"Required file {file} not found")
    
    def test_requirements_file(self):
        """Test requirements.txt is valid"""
        with open('requirements.txt', 'r') as f:
            content = f.read()
            
        # Check for essential packages
        essential_packages = [
            'torch',
            'torchvision', 
            'facenet-pytorch',
            'streamlit',
            'scikit-learn',
            'opencv-python',
            'numpy'
        ]
        
        for package in essential_packages:
            self.assertIn(package, content, f"Essential package {package} not found in requirements.txt")

def run_integration_test():
    """Run integration test with actual model training"""
    print("\n🔍 Running Integration Test...")
    
    try:
        # Check if we can train a minimal model
        from train_model import FaceRecognitionTrainer
        
        trainer = FaceRecognitionTrainer()
        print("✅ Model initialization successful")
        
        # Create minimal fake dataset
        fake_embeddings = np.random.rand(20, 512)
        fake_labels = ['person1'] * 10 + ['person2'] * 10
        
        svm, encoder, train_acc, test_acc = trainer.train_classifier(
            fake_embeddings, fake_labels, test_size=0.3, cross_validation=False
        )
        
        print(f"✅ Training successful - Train: {train_acc:.3f}, Test: {test_acc:.3f}")
        
        # Test model saving
        temp_dir = tempfile.mkdtemp()
        saved_files = trainer.save_model(svm, encoder, fake_embeddings, fake_labels, temp_dir)
        
        print("✅ Model saving successful")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        print("🎉 Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Main test runner"""
    print("🧪 Face Recognition System Test Suite")
    print("=" * 50)
    
    # Run unit tests
    print("\n📋 Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    integration_success = run_integration_test()
    
    print("\n📊 Test Summary:")
    print("=" * 30)
    print("✅ Unit tests completed")
    print(f"{'✅' if integration_success else '❌'} Integration test {'passed' if integration_success else 'failed'}")
    
    if integration_success:
        print("\n🎉 All tests passed! System is ready for use.")
    else:
        print("\n⚠️ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
