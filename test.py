import os
import logging
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_models(image_size=160, margin=20, pretrained='vggface2'):
    """
    Initialize MTCNN and InceptionResnetV1 models.

    Args:
        image_size (int): Size of the output image (default: 160).
        margin (int): Margin around detected faces (default: 20).
        pretrained (str): Pre-trained weights for InceptionResnetV1 (default: 'vggface2').

    Returns:
        tuple: (MTCNN, InceptionResnetV1, torch.device)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    mtcnn = MTCNN(image_size=image_size, margin=margin, keep_all=False, device=device)
    facenet = InceptionResnetV1(pretrained=pretrained).eval().to(device)
    
    return mtcnn, facenet, device

def load_dataset(data_dir):
    """
    Load images and labels from a directory structured as data_dir/person_name/image.jpg.

    Args:
        data_dir (str): Path to the dataset directory.

    Returns:
        tuple: (list of image paths, list of corresponding labels)
    """
    images, labels = [], []
    if not os.path.exists(data_dir):
        logging.error(f"Dataset directory {data_dir} does not exist")
        return images, labels
    
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append(os.path.join(person_dir, img_name))
                    labels.append(person)
    
    logging.info(f"Loaded {len(images)} images from {len(set(labels))} classes")
    return images, labels

def get_embedding(image_path, mtcnn, facenet, device, min_confidence=0.9):
    """
    Generate a face embedding from an image using MTCNN and InceptionResnetV1.

    Args:
        image_path (str): Path to the input image.
        mtcnn (MTCNN): MTCNN model for face detection.
        facenet (InceptionResnetV1): Pre-trained face embedding model.
        device (torch.device): Device to run the model on (CPU/GPU).
        min_confidence (float): Minimum confidence for face detection (default: 0.9).

    Returns:
        np.ndarray: 512-dimensional face embedding, or None if no face is detected.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        face, prob = mtcnn(img, return_prob=True)
        if face is None or prob < min_confidence:
            logging.warning(f"No face detected or low confidence ({prob}) in {image_path}")
            return None
        face = face.unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            embedding = facenet(face).cpu().numpy()
        return embedding.flatten()
    except Exception as e:
        logging.error(f"Failed to process {image_path}: {e}")
        return None

def get_batch_embeddings(image_paths, mtcnn, facenet, device, min_confidence=0.9):
    """
    Generate face embeddings for a batch of images.

    Args:
        image_paths (list): List of paths to input images.
        mtcnn (MTCNN): MTCNN model for face detection.
        facenet (InceptionResnetV1): Pre-trained face embedding model.
        device (torch.device): Device to run the model on (CPU/GPU).
        min_confidence (float): Minimum confidence for face detection (default: 0.9).

    Returns:
        tuple: (np.ndarray of embeddings, list of valid image paths)
    """
    faces, valid_paths = [], []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            face, prob = mtcnn(img, return_prob=True)
            if face is not None and prob >= min_confidence:
                faces.append(face)
                valid_paths.append(path)
        except Exception as e:
            logging.error(f"Failed to process {path}: {e}")
    
    if not faces:
        logging.warning("No valid faces detected in batch")
        return None, []
    
    faces = torch.stack(faces).to(device)
    with torch.no_grad():
        embeddings = facenet(faces).cpu().numpy()
    return embeddings, valid_paths

def train_model(embeddings, labels, svm_path='svm_model.pkl', encoder_path='label_encoder.pkl',
                embedding_file='embeddings.npy', label_file='labels.npy'):
    """
    Train SVM and save model, encoder, and raw data.
    """
    norm = Normalizer(norm='l2')
    embeddings = norm.transform(embeddings)
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    # Train SVM
    svm = SVC(kernel='linear', probability=True)
    svm.fit(embeddings, encoded_labels)

    # Save model and encoder
    joblib.dump(svm, svm_path)
    joblib.dump(encoder, encoder_path)
    logging.info(f"Model saved to {svm_path}, Encoder saved to {encoder_path}")

    # Save training data for future use (registration, retraining)
    np.save(embedding_file, embeddings)
    np.save(label_file, np.array(labels))  # Save raw labels
    logging.info(f"Saved embeddings to {embedding_file} and labels to {label_file}")

    return svm, encoder


def load_model(svm_path='svm_model.pkl', encoder_path='label_encoder.pkl'):
    """
    Load a pre-trained SVM classifier and LabelEncoder.

    Args:
        svm_path (str): Path to the SVM model file (default: 'svm_model.pkl').
        encoder_path (str): Path to the LabelEncoder file (default: 'label_encoder.pkl').

    Returns:
        tuple: (SVC, LabelEncoder)
    """
    try:
        svm = joblib.load(svm_path)
        encoder = joblib.load(encoder_path)
        logging.info(f"Loaded model from {svm_path} and encoder from {encoder_path}")
        return svm, encoder
    except Exception as e:
        logging.error(f"Failed to load model or encoder: {e}")
        return None, None

def predict_face(image_path, mtcnn, facenet, svm, encoder, device, min_confidence=0.9):
    """
    Predict the identity of a face in an image.

    Args:
        image_path (str): Path to the input image.
        mtcnn (MTCNN): MTCNN model for face detection.
        facenet (InceptionResnetV1): Pre-trained face embedding model.
        svm (SVC): Trained SVM classifier.
        encoder (LabelEncoder): Label encoder for class names.
        device (torch.device): Device to run the model on (CPU/GPU).
        min_confidence (float): Minimum confidence for face detection (default: 0.9).

    Returns:
        str: Predicted identity, or 'Unknown' if no face is detected or prediction fails.
    """
    embedding = get_embedding(image_path, mtcnn, facenet, device, min_confidence)
    if embedding is None:
        return 'Unknown'
    
    embedding = Normalizer(norm='l2').transform([embedding])
    pred = svm.predict(embedding)
    return encoder.inverse_transform(pred)[0]

def main(data_dir, test_size=0.3):
    """
    Main function to run the face recognition pipeline.

    Args:
        data_dir (str): Path to the dataset directory.
        test_size (float): Fraction of data to use for testing (default: 0.3).
    """
    # Setup models
    mtcnn, facenet, device = setup_models()
    
    # Load dataset
    images, labels = load_dataset(data_dir)
    if not images:
        logging.error("No images loaded. Exiting.")
        return
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, stratify=labels)
    
    # Generate training embeddings
    train_embeddings, valid_train_paths = get_batch_embeddings(X_train, mtcnn, facenet, device)
    if train_embeddings is None:
        logging.error("No valid training embeddings. Exiting.")
        return
    valid_train_labels = [y_train[X_train.index(p)] for p in valid_train_paths]
    
    # Train model
    svm, encoder = train_model(train_embeddings, valid_train_labels)
    
    # Evaluate on test set
    test_embeddings, valid_test_paths = get_batch_embeddings(X_test, mtcnn, facenet, device)
    if test_embeddings is not None:
        valid_test_labels = [y_test[X_test.index(p)] for p in valid_test_paths]
        test_embeddings = Normalizer(norm='l2').transform(test_embeddings)
        y_pred = svm.predict(test_embeddings)
        print(classification_report(encoder.transform(valid_test_labels), y_pred, target_names=encoder.classes_))
    
    # Example prediction
    sample_image = X_test[0] if X_test else None
    if sample_image:
        prediction = predict_face(sample_image, mtcnn, facenet, svm, encoder, device)
        logging.info(f"Predicted identity for {sample_image}: {prediction}")

if __name__ == "__main__":
    data_dir = "images dataset"
    main(data_dir)