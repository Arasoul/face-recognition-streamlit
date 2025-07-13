import streamlit as st
import logging
import torch
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import Normalizer
import joblib
import os
import tempfile
import io
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Page configuration
st.set_page_config(
    page_title="🎯 Face Recognition System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cyberpunk theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(45deg, #1a1a3a, #2e2e5c);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        border: 2px solid #4a90e2;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a1a, #1a1a3a);
    }
    
    .metric-container {
        background: rgba(26, 26, 58, 0.8);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #4a90e2;
        margin: 0.5rem 0;
    }
    
    .detection-box {
        border: 2px solid #00ffff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background: rgba(0, 255, 255, 0.1);
    }
    
    .unknown-box {
        border: 2px solid #ff3333;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background: rgba(255, 51, 51, 0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a3a, #2e2e5c);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load face recognition models with caching"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"🔧 Using device: {device}")
        
        mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)
        facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        st.success("✅ Models loaded successfully!")
        return mtcnn, facenet, device
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        return None, None, None

@st.cache_resource
def load_classifier():
    """Load trained SVM classifier with caching"""
    try:
        if os.path.exists('svm_model.pkl') and os.path.exists('label_encoder.pkl'):
            svm = joblib.load('svm_model.pkl')
            encoder = joblib.load('label_encoder.pkl')
            st.success("✅ Trained classifier loaded!")
            return svm, encoder
        else:
            st.warning("⚠️ No trained model found. Please train a model first.")
            return None, None
    except Exception as e:
        st.error(f"❌ Failed to load classifier: {e}")
        return None, None

def get_faces_and_embeddings(frame, mtcnn, facenet, device, min_conf=0.90):
    """Extract faces and their embeddings from frame"""
    try:
        if isinstance(frame, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            img = frame
            
        boxes, probs = mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            return []

        faces = []
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob >= min_conf:
                try:
                    box = [max(0, int(coord)) for coord in box]
                    face_tensor = mtcnn.extract(img, [box], save_path=None)
                    if face_tensor is None or len(face_tensor) == 0:
                        continue
                    face = face_tensor[0].unsqueeze(0).to(device)
                    with torch.no_grad():
                        embedding = facenet(face).cpu().numpy()
                    faces.append((box, embedding.flatten(), prob))
                except Exception as e:
                    st.warning(f"⚠️ Failed to process face {i}: {e}")
                    continue
        return faces
    except Exception as e:
        st.error(f"❌ Error in face detection: {e}")
        return []

def predict_faces(frame, mtcnn, facenet, svm, encoder, device):
    """Predict identities of faces in frame"""
    results = []
    if svm is None or encoder is None:
        return results

    faces = get_faces_and_embeddings(frame, mtcnn, facenet, device)
    if not faces:
        return results

    for box, embedding, conf in faces:
        norm_emb = Normalizer(norm='l2').transform([embedding])
        try:
            probs = svm.predict_proba(norm_emb)[0]
            pred = svm.predict(norm_emb)[0]
            identity = encoder.inverse_transform([pred])[0]
            confidence = probs[pred] * 100
            if confidence < 40:
                identity = 'Unknown'
                confidence = conf * 100
        except (ValueError, IndexError) as e:
            identity, confidence = 'Unknown', conf * 100
        except Exception as e:
            identity, confidence = 'Unknown', 0.0
        results.append((box, identity, confidence))
    return results

def draw_detections(img, results):
    """Draw detection boxes and labels on image"""
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for box, name, conf in results:
        # Color coding
        if name == "Unknown":
            box_color = "#ff3333"
            glow_color = "#ff6666"
            status_icon = "⚠️"
        else:
            box_color = "#00ffff"
            glow_color = "#66ffff"
            status_icon = "✅"
        
        # Draw bounding box with glow effect
        box_coords = [int(coord) for coord in box]
        
        # Glow effect
        for offset in range(3, 0, -1):
            glow_box = [
                box_coords[0] - offset, box_coords[1] - offset,
                box_coords[2] + offset, box_coords[3] + offset
            ]
            draw.rectangle(glow_box, outline=glow_color, width=1)
        
        # Main box
        draw.rectangle(box_coords, outline=box_color, width=3)
        
        # Text with background
        display_text = f"{status_icon} {name.upper()}"
        confidence_text = f"CONF: {conf:.1f}%"
        
        text_bbox = draw.textbbox((box_coords[0], box_coords[1] - 50), display_text, font=font)
        draw.rectangle([text_bbox[0]-5, text_bbox[1]-2, text_bbox[2]+5, text_bbox[3]+25], 
                     fill="#000000", outline=box_color)
        
        draw.text((box_coords[0], box_coords[1] - 45), display_text, fill=box_color, font=font)
        draw.text((box_coords[0], box_coords[1] - 25), confidence_text, fill=box_color, font=font)

    return img

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def process_folder_images(uploaded_files, mtcnn, facenet, svm, encoder, device):
    """Process multiple uploaded images"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
        
        # Process image
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        
        # Convert to BGR if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        detections = predict_faces(img_array, mtcnn, facenet, svm, encoder, device)
        
        results.append({
            'filename': uploaded_file.name,
            'image': img,
            'detections': detections,
            'face_count': len(detections)
        })
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 NEURAL FACE RECOGNITION MATRIX</h1>
        <p>AI-Powered Biometric Identification System</p>
        <p>🚀 Version 3.0 | 🧠 Neural Net: ONLINE | ⚡ Status: ACTIVE</p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    mtcnn, facenet, device = load_models()
    if mtcnn is None:
        st.error("❌ Failed to load models. Please check your installation.")
        return
    
    svm, encoder = load_classifier()

    # Sidebar
    st.sidebar.markdown("## 🎮 Control Matrix")
    
    mode = st.sidebar.selectbox(
        "🔍 Select Recognition Mode",
        ["📷 Webcam Recognition", "🖼️ Image Upload", "📁 Batch Processing", "➕ Register New Face"]
    )
    
    st.sidebar.markdown("---")
    
    # AI Status Panel
    st.sidebar.markdown("### 🤖 AI Status")
    status_container = st.sidebar.container()
    
    with status_container:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**NEURAL NET:**")
            st.markdown("**DETECTOR:**")
            st.markdown("**CLASSIFIER:**")
            st.markdown("**DEVICE:**")
        
        with col2:
            st.markdown("🟢 ACTIVE")
            st.markdown("🟢 READY")
            st.markdown("🟡 LOADED" if svm is not None else "🔴 NO MODEL")
            st.markdown(f"⚡ {device}".upper())

    # Main content area
    if mode == "📷 Webcam Recognition":
        st.markdown("### 🔍 Real-time Neural Scanner")
        
        camera_input = st.camera_input("📹 Activate Neural Scanner")
        
        if camera_input is not None:
            # Process camera image
            img = Image.open(camera_input)
            img_array = np.array(img)
            
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            results = predict_faces(img_array, mtcnn, facenet, svm, encoder, device)
            
            # Draw detections
            img_with_detections = draw_detections(img.copy(), results)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(img_with_detections, caption="🎯 Neural Scan Results", use_column_width=True)
            
            with col2:
                st.markdown("### 📊 Scan Results")
                
                if results:
                    for i, (box, name, conf) in enumerate(results):
                        if name == "Unknown":
                            st.markdown(f"""
                            <div class="unknown-box">
                                <h4>⚠️ Unknown Subject {i+1}</h4>
                                <p><strong>Confidence:</strong> {conf:.1f}%</p>
                                <p><strong>Status:</strong> UNIDENTIFIED</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="detection-box">
                                <h4>✅ {name}</h4>
                                <p><strong>Confidence:</strong> {conf:.1f}%</p>
                                <p><strong>Status:</strong> CONFIRMED</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("🔍 No faces detected in the scan")

    elif mode == "🖼️ Image Upload":
        st.markdown("### 🖼️ Target Image Analysis")
        
        uploaded_file = st.file_uploader(
            "📁 Upload target image for analysis",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Select an image file to analyze for faces"
        )
        
        if uploaded_file is not None:
            # Display original image
            img = Image.open(uploaded_file)
            
            with st.spinner("🔍 Analyzing target image..."):
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                results = predict_faces(img_array, mtcnn, facenet, svm, encoder, device)
                img_with_detections = draw_detections(img.copy(), results)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(img_with_detections, caption="🎯 Analysis Results", use_column_width=True)
            
            with col2:
                st.markdown("### 📊 Detection Summary")
                
                # Metrics
                known_faces = sum(1 for _, name, _ in results if name != "Unknown")
                unknown_faces = len(results) - known_faces
                
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("✅ Known Faces", known_faces)
                with col2b:
                    st.metric("⚠️ Unknown Faces", unknown_faces)
                
                # Detailed results
                if results:
                    for i, (box, name, conf) in enumerate(results):
                        if name == "Unknown":
                            st.error(f"⚠️ Unknown Subject {i+1}: {conf:.1f}%")
                        else:
                            st.success(f"✅ {name}: {conf:.1f}%")
                else:
                    st.warning("🔍 No faces detected")

    elif mode == "📁 Batch Processing":
        st.markdown("### 📁 Batch Image Processing")
        
        uploaded_files = st.file_uploader(
            "📂 Upload multiple images for batch processing",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Select multiple image files to process at once"
        )
        
        if uploaded_files:
            st.info(f"📊 Processing {len(uploaded_files)} images...")
            
            with st.spinner("🔄 Analyzing batch images..."):
                batch_results = process_folder_images(uploaded_files, mtcnn, facenet, svm, encoder, device)
            
            # Summary statistics
            total_faces = sum(result['face_count'] for result in batch_results)
            total_known = 0
            total_unknown = 0
            
            for result in batch_results:
                for _, name, _ in result['detections']:
                    if name == "Unknown":
                        total_unknown += 1
                    else:
                        total_known += 1
            
            # Display summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📁 Images Processed", len(uploaded_files))
            with col2:
                st.metric("👥 Total Faces", total_faces)
            with col3:
                st.metric("✅ Known Faces", total_known)
            with col4:
                st.metric("⚠️ Unknown Faces", total_unknown)
            
            # Detailed results
            st.markdown("### 🔍 Detailed Results")
            
            for result in batch_results:
                with st.expander(f"📄 {result['filename']} - {result['face_count']} face(s) detected"):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Draw detections on image
                        img_with_detections = draw_detections(result['image'].copy(), result['detections'])
                        st.image(img_with_detections, caption=f"Results for {result['filename']}")
                    
                    with col2:
                        if result['detections']:
                            for i, (box, name, conf) in enumerate(result['detections']):
                                if name == "Unknown":
                                    st.error(f"⚠️ Unknown Subject {i+1}: {conf:.1f}%")
                                else:
                                    st.success(f"✅ {name}: {conf:.1f}%")
                        else:
                            st.warning("🔍 No faces detected")

    elif mode == "➕ Register New Face":
        st.markdown("### ➕ Register New Identity")
        st.warning("🚧 Model training feature is not implemented in this Streamlit version.")
        st.info("💡 To register new faces, please use the desktop application (final9.py)")
        
        st.markdown("""
        ### 📝 Instructions for Registration:
        
        1. **Use Desktop App**: Run `python final9.py` for full registration capabilities
        2. **Prepare Images**: Collect 10-20 clear photos of the person
        3. **Good Lighting**: Ensure well-lit, front-facing photos
        4. **Various Angles**: Include slight variations in pose and expression
        5. **Train Model**: Use the registration feature to train the classifier
        6. **Deploy**: Copy the generated model files to this Streamlit app
        
        ### 📂 Required Model Files:
        - `svm_model.pkl` - Trained SVM classifier
        - `label_encoder.pkl` - Person name encoder
        - `embeddings.npy` - Face embeddings database
        - `labels.npy` - Corresponding labels
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #4a90e2;">
        <p>🚀 Powered by PyTorch FaceNet | 🧠 Neural Networks | ⚡ Streamlit</p>
        <p>© 2025 Face Recognition Matrix | Version 3.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
