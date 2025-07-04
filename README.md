# Face Recognition System

A sophisticated face recognition application built with Streamlit, PyTorch, and FaceNet.

## Features
- 📷 Webcam face recognition
- 🖼️ Image upload recognition  
- 📁 Batch image processing
- ➕ Face registration for new people
- 🎯 Multiple recognition modes

## Deployment

### Streamlit Cloud
1. Push this code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click!

### Local Development
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Requirements
- Python 3.8+
- CUDA (optional, for GPU acceleration)
- Pre-trained model files:
  - `svm_model.pkl`
  - `label_encoder.pkl`

## Camera Issues
If camera doesn't work locally:
- ✅ **Deploy to Streamlit Cloud** (camera works perfectly)
- 🔄 Use file upload as alternative
- 🌐 Run on HTTPS for better camera support

## Model Files
Ensure these files are in the project directory:
- `svm_model.pkl` - Trained SVM classifier
- `label_encoder.pkl` - Label encoder for person names
- `images dataset/` - Training images directory

## Usage
1. Select mode from sidebar
2. For webcam: Allow camera permission
3. For upload: Choose image files
4. View recognition results instantly

## Browser Compatibility
- ✅ Chrome (best)
- ✅ Edge
- ⚠️ Firefox (limited)
- ❌ Safari (not recommended)
