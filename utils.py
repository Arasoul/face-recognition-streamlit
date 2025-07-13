#!/usr/bin/env python3
"""
Utility functions for Face Recognition System

This module contains helper functions used across the application.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Union
import logging
import hashlib
import json
from datetime import datetime
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageUtils:
    """Utility functions for image processing"""
    
    @staticmethod
    def validate_image(image_path: str) -> bool:
        """Validate if file is a valid image"""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def resize_image(image: Union[np.ndarray, Image.Image], max_size: int = 1920) -> Union[np.ndarray, Image.Image]:
        """Resize image while maintaining aspect ratio"""
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:  # PIL Image
            w, h = image.size
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        return image
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better face detection"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    @staticmethod
    def create_thumbnail(image: Union[np.ndarray, Image.Image], size: Tuple[int, int] = (150, 150)) -> Image.Image:
        """Create thumbnail of image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Create thumbnail
        image.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Create square thumbnail with padding
        thumbnail = Image.new('RGB', size, (0, 0, 0))
        x = (size[0] - image.width) // 2
        y = (size[1] - image.height) // 2
        thumbnail.paste(image, (x, y))
        
        return thumbnail

class FileUtils:
    """Utility functions for file operations"""
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Get MD5 hash of file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error getting file hash: {e}")
            return ""
    
    @staticmethod
    def safe_filename(filename: str) -> str:
        """Create safe filename by removing invalid characters"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename.strip()
    
    @staticmethod
    def ensure_directory(directory: str) -> bool:
        """Ensure directory exists, create if not"""
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")
            return False
    
    @staticmethod
    def get_files_by_extension(directory: str, extensions: List[str]) -> List[str]:
        """Get all files with specific extensions in directory"""
        files = []
        for ext in extensions:
            pattern = f"**/*{ext}"
            files.extend(list(Path(directory).glob(pattern)))
        return [str(f) for f in files]

class MetricsUtils:
    """Utility functions for metrics and statistics"""
    
    @staticmethod
    def calculate_confidence_distribution(confidences: List[float]) -> dict:
        """Calculate confidence score distribution"""
        if not confidences:
            return {}
        
        return {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences),
            'q25': np.percentile(confidences, 25),
            'q75': np.percentile(confidences, 75)
        }
    
    @staticmethod
    def create_confusion_matrix_plot(y_true: List, y_pred: List, class_names: List[str], save_path: Optional[str] = None) -> str:
        """Create confusion matrix plot"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            # Save to temporary file
            save_path = tempfile.mktemp(suffix='.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    @staticmethod
    def create_training_history_plot(history: dict, save_path: Optional[str] = None) -> str:
        """Create training history plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        if 'train_accuracy' in history and 'val_accuracy' in history:
            ax1.plot(history['train_accuracy'], label='Training Accuracy')
            ax1.plot(history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
        
        # Loss plot
        if 'train_loss' in history and 'val_loss' in history:
            ax2.plot(history['train_loss'], label='Training Loss')
            ax2.plot(history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = tempfile.mktemp(suffix='.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path

class LoggingUtils:
    """Utility functions for logging and monitoring"""
    
    @staticmethod
    def setup_logger(name: str, log_file: str = None, level: str = 'INFO') -> logging.Logger:
        """Set up logger with file and console handlers"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_performance(func):
        """Decorator to log function execution time"""
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.info(f"Function {func.__name__} executed in {execution_time:.2f} seconds")
            return result
        return wrapper

class DataUtils:
    """Utility functions for data handling"""
    
    @staticmethod
    def save_results(results: dict, filename: str):
        """Save results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, datetime):
                serializable_results[key] = value.isoformat()
            else:
                serializable_results[key] = value
        
        try:
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    @staticmethod
    def load_results(filename: str) -> dict:
        """Load results from JSON file"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return {}
    
    @staticmethod
    def create_dataset_summary(dataset_dir: str) -> dict:
        """Create summary of dataset"""
        summary = {
            'total_persons': 0,
            'total_images': 0,
            'persons': {},
            'file_sizes': [],
            'image_dimensions': [],
            'created_at': datetime.now().isoformat()
        }
        
        if not os.path.exists(dataset_dir):
            return summary
        
        for person in os.listdir(dataset_dir):
            person_dir = os.path.join(dataset_dir, person)
            if os.path.isdir(person_dir):
                images = [f for f in os.listdir(person_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                summary['persons'][person] = {
                    'image_count': len(images),
                    'images': images
                }
                summary['total_images'] += len(images)
                
                # Analyze image properties
                for img_name in images[:5]:  # Sample first 5 images
                    img_path = os.path.join(person_dir, img_name)
                    try:
                        # File size
                        file_size = os.path.getsize(img_path)
                        summary['file_sizes'].append(file_size)
                        
                        # Image dimensions
                        with Image.open(img_path) as img:
                            summary['image_dimensions'].append(img.size)
                    except Exception:
                        continue
        
        summary['total_persons'] = len(summary['persons'])
        
        # Calculate statistics
        if summary['file_sizes']:
            summary['avg_file_size'] = np.mean(summary['file_sizes'])
            summary['total_dataset_size'] = sum(summary['file_sizes'])
        
        if summary['image_dimensions']:
            widths, heights = zip(*summary['image_dimensions'])
            summary['avg_image_size'] = (np.mean(widths), np.mean(heights))
        
        return summary

# Convenience functions
def quick_face_detection(image_path: str, mtcnn, min_confidence: float = 0.9) -> List[Tuple]:
    """Quick face detection for single image"""
    try:
        img = Image.open(image_path).convert('RGB')
        boxes, probs = mtcnn.detect(img)
        
        if boxes is None:
            return []
        
        faces = []
        for box, prob in zip(boxes, probs):
            if prob >= min_confidence:
                faces.append((box, prob))
        
        return faces
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return []

def create_face_grid(images: List[Image.Image], grid_size: Tuple[int, int] = (5, 4)) -> Image.Image:
    """Create grid of face images"""
    rows, cols = grid_size
    img_width, img_height = 150, 150
    
    grid_width = cols * img_width
    grid_height = rows * img_height
    
    grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    for i, img in enumerate(images[:rows * cols]):
        if img is None:
            continue
            
        # Resize image
        img_resized = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
        
        # Calculate position
        row = i // cols
        col = i % cols
        x = col * img_width
        y = row * img_height
        
        grid_image.paste(img_resized, (x, y))
    
    return grid_image

if __name__ == "__main__":
    # Example usage
    print("🔧 Face Recognition Utilities")
    print("=" * 30)
    
    # Test image utilities
    print("Testing image utilities...")
    
    # Test file utilities
    print("Testing file utilities...")
    
    # Test metrics utilities
    print("Testing metrics utilities...")
    
    print("✅ All utilities tested successfully!")
