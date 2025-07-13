# 🎯 Face Recognition API Documentation

## Overview

This document describes the Face Recognition System's API and integration capabilities.

## Core Components

### 1. Model Training API

```python
from train_model import FaceRecognitionTrainer

# Initialize trainer
trainer = FaceRecognitionTrainer()

# Load dataset
images, labels = trainer.load_dataset("dataset/", min_images_per_person=5)

# Extract embeddings
embeddings, valid_paths, failed_paths = trainer.extract_batch_embeddings(images)

# Train classifier
svm, encoder, train_acc, test_acc = trainer.train_classifier(embeddings, labels)

# Save model
saved_files = trainer.save_model(svm, encoder, embeddings, labels)
```

### 2. Recognition API

```python
from utils import FaceRecognitionSystem

# Initialize system
system = FaceRecognitionSystem()

# Process single image
results = system.process_image("path/to/image.jpg")

# Process batch
batch_results = system.process_batch(["img1.jpg", "img2.jpg"])

# Real-time processing
frame_results = system.process_frame(frame_array)
```

### 3. Streamlit Integration

```python
import streamlit as st
from streamlit_app import load_models, predict_faces

# Load models
mtcnn, facenet, device = load_models()
svm, encoder = load_classifier()

# Process uploaded image
uploaded_file = st.file_uploader("Upload Image")
if uploaded_file:
    results = predict_faces(image_array, mtcnn, facenet, svm, encoder, device)
```

## REST API Endpoints (Future)

### Authentication
```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "secure_password"
}
```

### Face Recognition
```http
POST /api/recognize
Authorization: Bearer {token}
Content-Type: multipart/form-data

{
  "image": <binary_image_data>,
  "confidence_threshold": 0.8
}
```

Response:
```json
{
  "faces": [
    {
      "name": "John Doe",
      "confidence": 0.95,
      "bounding_box": [100, 150, 200, 250],
      "timestamp": "2025-07-13T12:00:00Z"
    }
  ],
  "processing_time": 0.25
}
```

### Batch Processing
```http
POST /api/batch
Authorization: Bearer {token}
Content-Type: multipart/form-data

{
  "images": [<binary_data_1>, <binary_data_2>],
  "options": {
    "confidence_threshold": 0.8,
    "return_embeddings": false
  }
}
```

### Model Training
```http
POST /api/train
Authorization: Bearer {token}
Content-Type: multipart/form-data

{
  "dataset": <zip_file_with_person_folders>,
  "options": {
    "min_images_per_person": 5,
    "test_size": 0.2
  }
}
```

## Configuration

### Environment Variables

```bash
# Model Configuration
FACE_DETECTION_CONFIDENCE=0.9
RECOGNITION_CONFIDENCE=0.4
IMAGE_SIZE=160
BATCH_SIZE=32

# Performance
USE_GPU=true
MAX_CONCURRENT_REQUESTS=10
CACHE_EMBEDDINGS=true

# Security
SECRET_KEY=your_secret_key
ALLOWED_ORIGINS=["http://localhost:3000"]
MAX_FILE_SIZE=10MB

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost/facedb
REDIS_URL=redis://localhost:6379
```

### Model Files Structure

```
models/
├── svm_model.pkl          # Trained SVM classifier
├── label_encoder.pkl      # Label encoder
├── embeddings.npy         # Face embeddings
├── labels.npy            # Corresponding labels
├── training_stats.json   # Training statistics
└── config.json          # Model configuration
```

## Error Handling

### Common Error Codes

- `400` - Bad Request (invalid image format)
- `401` - Unauthorized (invalid token)
- `413` - Payload Too Large (image size exceeded)
- `422` - Unprocessable Entity (no faces detected)
- `500` - Internal Server Error

### Error Response Format

```json
{
  "error": {
    "code": "NO_FACES_DETECTED",
    "message": "No faces found in the uploaded image",
    "details": {
      "confidence_threshold": 0.9,
      "image_size": [640, 480]
    }
  },
  "timestamp": "2025-07-13T12:00:00Z"
}
```

## Performance Considerations

### Optimization Tips

1. **GPU Acceleration**: Use CUDA for 5-10x speedup
2. **Batch Processing**: Process multiple images together
3. **Image Preprocessing**: Resize images to optimal size
4. **Caching**: Cache embeddings for known faces
5. **Load Balancing**: Use multiple workers for high load

### Benchmarks

| Operation | CPU (avg) | GPU (avg) | Memory |
|-----------|-----------|-----------|---------|
| Face Detection | 200ms | 50ms | 500MB |
| Embedding Extraction | 100ms | 20ms | 300MB |
| Classification | 5ms | 2ms | 50MB |
| Full Pipeline | 305ms | 72ms | 850MB |

## Integration Examples

### Python Client

```python
import requests
import base64

def recognize_face(image_path, api_url, token):
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = requests.post(
        f"{api_url}/api/recognize",
        headers={"Authorization": f"Bearer {token}"},
        json={"image": image_data}
    )
    
    return response.json()
```

### JavaScript Client

```javascript
async function recognizeFace(imageFile, apiUrl, token) {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    const response = await fetch(`${apiUrl}/api/recognize`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`
        },
        body: formData
    });
    
    return await response.json();
}
```

### cURL Example

```bash
curl -X POST \
  http://localhost:8501/api/recognize \
  -H "Authorization: Bearer your_token" \
  -F "image=@photo.jpg" \
  -F "confidence_threshold=0.8"
```

## Deployment Guide

### Docker Deployment

```bash
# Build image
docker build -t face-recognition .

# Run container
docker run -p 8501:8501 face-recognition

# Using docker-compose
docker-compose up -d
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-recognition
spec:
  replicas: 3
  selector:
    matchLabels:
      app: face-recognition
  template:
    metadata:
      labels:
        app: face-recognition
    spec:
      containers:
      - name: app
        image: face-recognition:latest
        ports:
        - containerPort: 8501
        env:
        - name: USE_GPU
          value: "false"
```

## Security Best Practices

1. **Authentication**: Always use API tokens
2. **HTTPS**: Use SSL/TLS in production
3. **Rate Limiting**: Implement request rate limiting
4. **Input Validation**: Validate all uploaded images
5. **Data Privacy**: Don't store biometric data unnecessarily
6. **Audit Logging**: Log all recognition attempts

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use CPU mode
   - Restart application

2. **Poor Recognition Accuracy**
   - Add more training images
   - Improve image quality
   - Retrain model

3. **Slow Performance**
   - Enable GPU acceleration
   - Optimize image sizes
   - Use caching

For more help, check the [GitHub Issues](https://github.com/Arasoul/face-recognition-streamlit/issues) or contact support.
