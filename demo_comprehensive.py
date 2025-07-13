#!/usr/bin/env python3
"""
Comprehensive Demo for Face Recognition System

This script demonstrates all the capabilities of the face recognition system
and helps users understand how to use different features.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from utils import ImageUtils, FileUtils, MetricsUtils, DataUtils
from train_model import FaceRecognitionTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognitionDemo:
    """Comprehensive demo of face recognition system"""
    
    def __init__(self):
        self.config = get_config()
        self.trainer = None
        self.demo_results = {}
        
    def setup_demo_environment(self):
        """Set up demo environment with sample data"""
        print("🚀 Setting up demo environment...")
        
        # Create demo directories
        demo_dirs = [
            "demo_dataset",
            "demo_output",
            "demo_logs"
        ]
        
        for dir_name in demo_dirs:
            FileUtils.ensure_directory(dir_name)
        
        print("✅ Demo environment ready")
    
    def demonstrate_model_loading(self):
        """Demonstrate model loading and initialization"""
        print("\n🧠 Demonstrating Model Loading...")
        print("=" * 50)
        
        try:
            start_time = time.time()
            self.trainer = FaceRecognitionTrainer()
            load_time = time.time() - start_time
            
            print(f"✅ Models loaded successfully in {load_time:.2f} seconds")
            print(f"🔧 Device: {self.trainer.device}")
            print(f"📊 Embedding size: {self.trainer.embedding_size}")
            
            self.demo_results['model_load_time'] = load_time
            self.demo_results['device'] = str(self.trainer.device)
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
        
        return True
    
    def demonstrate_dataset_analysis(self):
        """Demonstrate dataset analysis capabilities"""
        print("\n📊 Demonstrating Dataset Analysis...")
        print("=" * 50)
        
        dataset_dir = "images dataset"
        if not os.path.exists(dataset_dir):
            print("⚠️ No dataset found. Creating sample structure...")
            self._create_sample_dataset_structure()
        
        # Analyze dataset
        summary = DataUtils.create_dataset_summary(dataset_dir)
        
        print(f"📁 Dataset Summary:")
        print(f"   • Total persons: {summary['total_persons']}")
        print(f"   • Total images: {summary['total_images']}")
        print(f"   • Average file size: {summary.get('avg_file_size', 0)/1024:.1f} KB")
        
        if summary['persons']:
            print(f"   • Persons detected:")
            for person, info in summary['persons'].items():
                print(f"     - {person}: {info['image_count']} images")
        
        # Save summary
        DataUtils.save_results(summary, "demo_output/dataset_summary.json")
        self.demo_results['dataset_summary'] = summary
        
        return summary['total_images'] > 0
    
    def demonstrate_face_detection(self):
        """Demonstrate face detection capabilities"""
        print("\n🔍 Demonstrating Face Detection...")
        print("=" * 50)
        
        if not self.trainer:
            print("❌ Models not loaded")
            return False
        
        # Look for sample images
        sample_dir = "sample_images"
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        else:
            sample_files = []
        
        if not sample_files:
            print("⚠️ No sample images found. Please add images to 'sample_images/' directory")
            return False
        
        detection_results = []
        
        for img_file in sample_files[:3]:  # Test first 3 images
            img_path = os.path.join(sample_dir, img_file)
            print(f"🖼️ Processing: {img_file}")
            
            try:
                start_time = time.time()
                faces = self.trainer.extract_embedding(img_path)
                detection_time = time.time() - start_time
                
                if faces is not None:
                    print(f"   ✅ Face detected in {detection_time:.3f} seconds")
                    detection_results.append({
                        'file': img_file,
                        'detected': True,
                        'time': detection_time
                    })
                else:
                    print(f"   ❌ No face detected")
                    detection_results.append({
                        'file': img_file,
                        'detected': False,
                        'time': detection_time
                    })
                    
            except Exception as e:
                print(f"   ❌ Error processing {img_file}: {e}")
        
        # Calculate statistics
        successful_detections = [r for r in detection_results if r['detected']]
        if successful_detections:
            avg_time = np.mean([r['time'] for r in successful_detections])
            print(f"\n📊 Detection Statistics:")
            print(f"   • Success rate: {len(successful_detections)}/{len(detection_results)} ({len(successful_detections)/len(detection_results)*100:.1f}%)")
            print(f"   • Average detection time: {avg_time:.3f} seconds")
        
        self.demo_results['detection_results'] = detection_results
        return len(successful_detections) > 0
    
    def demonstrate_training_pipeline(self):
        """Demonstrate training pipeline with mock data"""
        print("\n🎓 Demonstrating Training Pipeline...")
        print("=" * 50)
        
        # Check if we have real data
        dataset_dir = "images dataset"
        if os.path.exists(dataset_dir):
            image_paths, labels = self.trainer.load_dataset(dataset_dir, min_images_per_person=2)
            
            if len(image_paths) >= 4:  # Minimum for training demo
                print("📁 Using real dataset for training demo")
                return self._demo_real_training(image_paths, labels)
        
        # Use mock data if no real dataset
        print("🎭 Using mock data for training demonstration")
        return self._demo_mock_training()
    
    def _demo_real_training(self, image_paths, labels):
        """Demo with real dataset"""
        try:
            print(f"📊 Dataset: {len(image_paths)} images, {len(set(labels))} persons")
            
            # Extract embeddings
            print("🔄 Extracting embeddings...")
            embeddings, valid_paths, failed_paths = self.trainer.extract_batch_embeddings(
                image_paths[:10], min_confidence=0.8  # Limit to 10 images for demo
            )
            
            if len(embeddings) < 4:
                print("⚠️ Not enough valid embeddings for training demo")
                return False
            
            # Update labels for valid paths
            valid_labels = [labels[image_paths.index(path)] for path in valid_paths]
            
            print(f"✅ Extracted {len(embeddings)} valid embeddings")
            
            # Train classifier
            print("🧠 Training classifier...")
            svm, encoder, train_acc, test_acc = self.trainer.train_classifier(
                embeddings, valid_labels, test_size=0.3, cross_validation=False
            )
            
            print(f"📊 Training Results:")
            print(f"   • Training accuracy: {train_acc:.3f}")
            print(f"   • Test accuracy: {test_acc:.3f}")
            
            self.demo_results['training_results'] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'num_samples': len(embeddings),
                'num_classes': len(set(valid_labels))
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Training demo failed: {e}")
            return False
    
    def _demo_mock_training(self):
        """Demo with mock data"""
        try:
            # Create mock embeddings
            print("🎭 Creating mock training data...")
            np.random.seed(42)  # For reproducible results
            
            # 3 classes, 10 samples each
            embeddings = []
            labels = []
            
            for i in range(3):
                # Create cluster of embeddings for each person
                center = np.random.randn(512)
                for j in range(10):
                    noise = np.random.randn(512) * 0.1
                    embedding = center + noise
                    embeddings.append(embedding)
                    labels.append(f'person_{i+1}')
            
            embeddings = np.array(embeddings)
            print(f"✅ Created {len(embeddings)} mock embeddings for {len(set(labels))} persons")
            
            # Train classifier
            print("🧠 Training on mock data...")
            svm, encoder, train_acc, test_acc = self.trainer.train_classifier(
                embeddings, labels, test_size=0.3, cross_validation=True
            )
            
            print(f"📊 Mock Training Results:")
            print(f"   • Training accuracy: {train_acc:.3f}")
            print(f"   • Test accuracy: {test_acc:.3f}")
            
            self.demo_results['mock_training_results'] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'num_samples': len(embeddings),
                'num_classes': len(set(labels))
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Mock training failed: {e}")
            return False
    
    def demonstrate_streamlit_features(self):
        """Demonstrate Streamlit app features"""
        print("\n🌐 Demonstrating Streamlit Features...")
        print("=" * 50)
        
        try:
            import streamlit as st
            print("✅ Streamlit is available")
            
            # Check if streamlit app can be imported
            import streamlit_app
            print("✅ Streamlit app can be imported")
            
            print("🚀 To run the web interface:")
            print("   streamlit run streamlit_app.py")
            
            return True
            
        except ImportError as e:
            print(f"❌ Streamlit not available: {e}")
            return False
        except Exception as e:
            print(f"⚠️ Streamlit app check failed: {e}")
            return False
    
    def demonstrate_desktop_app(self):
        """Demonstrate desktop app features"""
        print("\n🖥️ Demonstrating Desktop Application...")
        print("=" * 50)
        
        try:
            # Check if desktop app can be imported
            import final9
            print("✅ Desktop app can be imported")
            
            print("🚀 To run the desktop interface:")
            print("   python final9.py")
            
            return True
            
        except ImportError as e:
            print(f"❌ Desktop app dependencies missing: {e}")
            return False
        except Exception as e:
            print(f"⚠️ Desktop app check failed: {e}")
            return False
    
    def _create_sample_dataset_structure(self):
        """Create sample dataset structure"""
        dataset_dir = "images dataset"
        sample_persons = ["john_doe", "jane_smith", "alex_johnson"]
        
        for person in sample_persons:
            person_dir = os.path.join(dataset_dir, person)
            FileUtils.ensure_directory(person_dir)
            
            # Create README
            readme_path = os.path.join(person_dir, "README.txt")
            with open(readme_path, 'w') as f:
                f.write(f"Add images of {person} here.\n")
                f.write("Recommended: 5-10 clear, front-facing photos.\n")
    
    def generate_demo_report(self):
        """Generate comprehensive demo report"""
        print("\n📊 Generating Demo Report...")
        print("=" * 50)
        
        report = {
            'demo_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform
            },
            'config': {
                'model': self.config.model.__dict__,
                'ui': self.config.ui.__dict__,
                'processing': self.config.processing.__dict__
            },
            'results': self.demo_results
        }
        
        # Save report
        report_path = "demo_output/demo_report.json"
        DataUtils.save_results(report, report_path)
        
        # Create summary
        print("📋 Demo Summary:")
        print(f"   • Report saved to: {report_path}")
        print(f"   • Total tests run: {len(self.demo_results)}")
        
        # Calculate success rate
        successful_tests = sum(1 for key in self.demo_results.keys() 
                             if key.endswith('_results') or key.endswith('_time'))
        print(f"   • Successful operations: {successful_tests}")
        
        return report_path
    
    def run_full_demo(self):
        """Run complete demonstration"""
        print("🎯 Face Recognition System - Full Demo")
        print("=" * 60)
        
        self.setup_demo_environment()
        
        # Run all demonstrations
        demos = [
            ("Model Loading", self.demonstrate_model_loading),
            ("Dataset Analysis", self.demonstrate_dataset_analysis),
            ("Face Detection", self.demonstrate_face_detection),
            ("Training Pipeline", self.demonstrate_training_pipeline),
            ("Streamlit Features", self.demonstrate_streamlit_features),
            ("Desktop App", self.demonstrate_desktop_app)
        ]
        
        results = {}
        for demo_name, demo_func in demos:
            print(f"\n{'='*20} {demo_name} {'='*20}")
            try:
                results[demo_name] = demo_func()
            except Exception as e:
                print(f"❌ {demo_name} failed: {e}")
                results[demo_name] = False
        
        # Generate report
        report_path = self.generate_demo_report()
        
        # Final summary
        print(f"\n🎉 Demo Completed!")
        print(f"📊 Results:")
        for demo_name, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {demo_name}")
        
        successful_demos = sum(results.values())
        total_demos = len(results)
        print(f"\n📈 Success Rate: {successful_demos}/{total_demos} ({successful_demos/total_demos*100:.1f}%)")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Face Recognition System Demo')
    parser.add_argument('--quick', action='store_true', help='Run quick demo')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    demo = FaceRecognitionDemo()
    
    if args.quick:
        # Quick demo - just test imports and basic functionality
        print("⚡ Quick Demo Mode")
        success = demo.demonstrate_model_loading()
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")
    else:
        # Full demo
        demo.run_full_demo()

if __name__ == "__main__":
    main()
