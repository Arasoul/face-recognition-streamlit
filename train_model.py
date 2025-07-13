#!/usr/bin/env python3
"""
Face Recognition Model Training Script

This script trains a face recognition model using FaceNet embeddings and SVM classifier.
It creates all the necessary files required by the main applications:
- svm_model.pkl: Trained SVM classifier
- label_encoder.pkl: Label encoder for person names
- embeddings.npy: Face embeddings database
- labels.npy: Corresponding labels

Usage:
    python train_model.py --data_dir "images dataset" --min_images 5
    
Directory Structure:
    images dataset/
    ├── person1/
    │   ├── 0001.jpg
    │   ├── 0002.jpg
    │   └── ...
    ├── person2/
    │   ├── 0001.jpg
    │   └── ...
    └── ...
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class FaceRecognitionTrainer:
    """Face Recognition Model Trainer"""
    
    def __init__(self, image_size=160, margin=20, pretrained='vggface2'):
        """Initialize the trainer with models"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"🔧 Using device: {self.device}")
        
        self.mtcnn = MTCNN(
            image_size=image_size, 
            margin=margin, 
            keep_all=False, 
            device=self.device,
            post_process=False
        )
        
        self.facenet = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)
        logging.info("✅ Models initialized successfully")
        
        self.embedding_size = 512
        self.training_stats = {}
    
    def load_dataset(self, data_dir, min_images_per_person=5):
        """
        Load images and labels from directory structure
        
        Args:
            data_dir (str): Path to dataset directory
            min_images_per_person (int): Minimum images required per person
            
        Returns:
            tuple: (image_paths, labels)
        """
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory {data_dir} does not exist")
        
        images, labels = [], []
        person_counts = {}
        
        for person in os.listdir(data_dir):
            person_dir = os.path.join(data_dir, person)
            if not os.path.isdir(person_dir):
                continue
                
            person_images = []
            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    person_images.append(os.path.join(person_dir, img_name))
            
            if len(person_images) >= min_images_per_person:
                images.extend(person_images)
                labels.extend([person] * len(person_images))
                person_counts[person] = len(person_images)
                logging.info(f"✅ {person}: {len(person_images)} images")
            else:
                logging.warning(f"⚠️  {person}: Only {len(person_images)} images (minimum {min_images_per_person} required)")
        
        self.training_stats['person_counts'] = person_counts
        self.training_stats['total_images'] = len(images)
        self.training_stats['total_persons'] = len(person_counts)
        
        logging.info(f"📊 Dataset loaded: {len(images)} images from {len(person_counts)} persons")
        return images, labels
    
    def extract_embedding(self, image_path, min_confidence=0.9):
        """
        Extract face embedding from single image
        
        Args:
            image_path (str): Path to image
            min_confidence (float): Minimum detection confidence
            
        Returns:
            np.ndarray or None: Face embedding or None if failed
        """
        try:
            img = Image.open(image_path).convert('RGB')
            
            # Detect face
            face, prob = self.mtcnn(img, return_prob=True)
            
            if face is None or prob < min_confidence:
                logging.warning(f"❌ No face detected in {os.path.basename(image_path)} (conf: {prob:.3f})")
                return None
            
            # Generate embedding
            face = face.unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.facenet(face).cpu().numpy()
            
            return embedding.flatten()
            
        except Exception as e:
            logging.error(f"❌ Failed to process {image_path}: {e}")
            return None
    
    def extract_batch_embeddings(self, image_paths, min_confidence=0.9, batch_size=32):
        """
        Extract embeddings from batch of images with progress bar
        
        Args:
            image_paths (list): List of image paths
            min_confidence (float): Minimum detection confidence
            batch_size (int): Processing batch size
            
        Returns:
            tuple: (embeddings, valid_paths, failed_paths)
        """
        embeddings = []
        valid_paths = []
        failed_paths = []
        
        logging.info(f"🔄 Processing {len(image_paths)} images...")
        
        with tqdm(total=len(image_paths), desc="Extracting embeddings") as pbar:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                
                for path in batch_paths:
                    embedding = self.extract_embedding(path, min_confidence)
                    if embedding is not None:
                        embeddings.append(embedding)
                        valid_paths.append(path)
                    else:
                        failed_paths.append(path)
                    
                    pbar.update(1)
        
        if embeddings:
            embeddings = np.array(embeddings)
            logging.info(f"✅ Successfully processed {len(embeddings)}/{len(image_paths)} images")
        else:
            logging.error("❌ No valid embeddings extracted!")
            
        return embeddings, valid_paths, failed_paths
    
    def train_classifier(self, embeddings, labels, test_size=0.2, cross_validation=True):
        """
        Train SVM classifier with evaluation
        
        Args:
            embeddings (np.ndarray): Face embeddings
            labels (list): Corresponding labels
            test_size (float): Test set ratio
            cross_validation (bool): Perform cross-validation
            
        Returns:
            tuple: (svm_model, label_encoder, train_score, test_score)
        """
        logging.info("🧠 Training SVM classifier...")
        
        # Normalize embeddings
        normalizer = Normalizer(norm='l2')
        embeddings_norm = normalizer.fit_transform(embeddings)
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings_norm, encoded_labels, 
            test_size=test_size, 
            stratify=encoded_labels,
            random_state=42
        )
        
        # Train SVM
        svm_model = SVC(
            kernel='linear', 
            probability=True, 
            C=1.0,
            random_state=42
        )
        svm_model.fit(X_train, y_train)
        
        # Evaluate
        train_score = svm_model.score(X_train, y_train)
        test_score = svm_model.score(X_test, y_test)
        
        logging.info(f"📊 Training accuracy: {train_score:.3f}")
        logging.info(f"📊 Test accuracy: {test_score:.3f}")
        
        # Cross-validation
        if cross_validation and len(np.unique(encoded_labels)) > 2:
            cv_scores = cross_val_score(svm_model, embeddings_norm, encoded_labels, cv=3)
            logging.info(f"📊 Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            self.training_stats['cv_scores'] = cv_scores.tolist()
        
        # Detailed evaluation
        y_pred = svm_model.predict(X_test)
        class_names = label_encoder.classes_
        
        logging.info("\n📊 Classification Report:")
        report = classification_report(y_test, y_pred, target_names=class_names)
        print(report)
        
        # Store stats
        self.training_stats.update({
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'num_classes': len(class_names),
            'class_names': class_names.tolist(),
            'embedding_dim': embeddings.shape[1],
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        })
        
        return svm_model, label_encoder, train_score, test_score
    
    def save_model(self, svm_model, label_encoder, embeddings, labels, 
                   output_dir='.', model_name='face_recognition'):
        """
        Save trained model and data
        
        Args:
            svm_model: Trained SVM model
            label_encoder: Label encoder
            embeddings: Training embeddings
            labels: Training labels
            output_dir: Output directory
            model_name: Base name for model files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model files
        svm_path = os.path.join(output_dir, 'svm_model.pkl')
        encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
        embeddings_path = os.path.join(output_dir, 'embeddings.npy')
        labels_path = os.path.join(output_dir, 'labels.npy')
        stats_path = os.path.join(output_dir, 'training_stats.json')
        
        # Save model and encoder
        joblib.dump(svm_model, svm_path)
        joblib.dump(label_encoder, encoder_path)
        
        # Save training data
        np.save(embeddings_path, embeddings)
        np.save(labels_path, np.array(labels))
        
        # Save training statistics
        self.training_stats['training_date'] = datetime.now().isoformat()
        self.training_stats['model_version'] = '3.0'
        
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        logging.info(f"💾 Model saved to: {svm_path}")
        logging.info(f"💾 Encoder saved to: {encoder_path}")
        logging.info(f"💾 Embeddings saved to: {embeddings_path}")
        logging.info(f"💾 Labels saved to: {labels_path}")
        logging.info(f"💾 Statistics saved to: {stats_path}")
        
        return {
            'svm_model': svm_path,
            'label_encoder': encoder_path,
            'embeddings': embeddings_path,
            'labels': labels_path,
            'stats': stats_path
        }
    
    def create_visualization(self, embeddings, labels, output_dir='.'):
        """Create visualization of embeddings using t-SNE"""
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            
            logging.info("📊 Creating embedding visualization...")
            
            # Reduce dimensionality
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            unique_labels = list(set(labels))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors[i]], label=label, alpha=0.7, s=50)
            
            plt.title('Face Embeddings Visualization (t-SNE)', fontsize=16)
            plt.xlabel('t-SNE Component 1', fontsize=12)
            plt.ylabel('t-SNE Component 2', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            viz_path = os.path.join(output_dir, 'embeddings_visualization.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"📊 Visualization saved to: {viz_path}")
            
        except ImportError:
            logging.warning("⚠️ Visualization requires scikit-learn and matplotlib")
        except Exception as e:
            logging.error(f"❌ Failed to create visualization: {e}")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train Face Recognition Model')
    parser.add_argument('--data_dir', default='images dataset', 
                       help='Directory containing training images')
    parser.add_argument('--output_dir', default='.', 
                       help='Output directory for model files')
    parser.add_argument('--min_images', type=int, default=5,
                       help='Minimum images per person')
    parser.add_argument('--min_confidence', type=float, default=0.9,
                       help='Minimum face detection confidence')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set ratio')
    parser.add_argument('--visualize', action='store_true',
                       help='Create embeddings visualization')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = FaceRecognitionTrainer()
        
        # Load dataset
        logging.info("📁 Loading dataset...")
        image_paths, labels = trainer.load_dataset(args.data_dir, args.min_images)
        
        if not image_paths:
            logging.error("❌ No images found. Please check your dataset directory.")
            return
        
        # Extract embeddings
        embeddings, valid_paths, failed_paths = trainer.extract_batch_embeddings(
            image_paths, args.min_confidence
        )
        
        if len(embeddings) == 0:
            logging.error("❌ No valid embeddings extracted. Training failed.")
            return
        
        # Update labels for valid paths only
        valid_labels = [labels[image_paths.index(path)] for path in valid_paths]
        
        # Train classifier
        svm_model, label_encoder, train_acc, test_acc = trainer.train_classifier(
            embeddings, valid_labels, args.test_size
        )
        
        # Save model
        saved_files = trainer.save_model(
            svm_model, label_encoder, embeddings, valid_labels, args.output_dir
        )
        
        # Create visualization
        if args.visualize:
            trainer.create_visualization(embeddings, valid_labels, args.output_dir)
        
        # Summary
        logging.info("\n🎉 Training completed successfully!")
        logging.info(f"📊 Final Statistics:")
        logging.info(f"   • Persons: {len(set(valid_labels))}")
        logging.info(f"   • Images: {len(embeddings)}")
        logging.info(f"   • Training accuracy: {train_acc:.3f}")
        logging.info(f"   • Test accuracy: {test_acc:.3f}")
        logging.info(f"   • Failed images: {len(failed_paths)}")
        
        if failed_paths:
            logging.warning(f"⚠️ {len(failed_paths)} images failed processing:")
            for path in failed_paths[:5]:  # Show first 5
                logging.warning(f"   • {os.path.basename(path)}")
            if len(failed_paths) > 5:
                logging.warning(f"   • ... and {len(failed_paths)-5} more")
        
        logging.info("\n🚀 Model is ready for use with final9.py and streamlit_app.py!")
        
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
