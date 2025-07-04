import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import Normalizer
import joblib
import os
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import tempfile
import time
import random

# Configure page
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #4CAF50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .unknown-box {
        background-color: #fff2f2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def setup_models(image_size=160, margin=20, pretrained='vggface2'):
    """Initialize and cache the face detection and recognition models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=image_size, margin=margin, keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained=pretrained).eval().to(device)
    return mtcnn, facenet, device

@st.cache_resource
def load_trained_model(svm_path='svm_model.pkl', encoder_path='label_encoder.pkl'):
    """Load the trained SVM model and label encoder"""
    try:
        if os.path.exists(svm_path) and os.path.exists(encoder_path):
            svm = joblib.load(svm_path)
            encoder = joblib.load(encoder_path)
            return svm, encoder
        else:
            st.warning("⚠️ Pre-trained model files not found. Please ensure svm_model.pkl and label_encoder.pkl are in the current directory.")
            return None, None
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None

def get_faces_and_embeddings(frame, mtcnn, facenet, device, min_conf=0.90):
    """Extract faces and their embeddings from a frame"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, probs = mtcnn.detect(img)
    
    if boxes is None or len(boxes) == 0:
        return []

    faces = []
    for i, (box, prob) in enumerate(zip(boxes, probs)):
        if prob >= min_conf:
            face = mtcnn.extract(img, [box], save_path=None)[0].unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = facenet(face).cpu().numpy()
            faces.append((box, embedding.flatten(), prob))
    
    return faces

def predict_faces(frame, mtcnn, facenet, svm, encoder, device):
    """Predict identities of faces in a frame"""
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
        except:
            identity, confidence = 'Unknown', 0.0
        results.append((box, identity, confidence))
    
    return results

def draw_predictions(image, results):
    """Draw bounding boxes and labels on the image"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for box, name, conf in results:
        color = "red" if name == "Unknown" else "lime"
        draw.rectangle(box.tolist(), outline=color, width=3)
        text = f"{name} ({conf:.1f}%)" if name != "Unknown" else "Unknown"
        draw.text((box[0], box[1] - 25), text, fill=color, font=font)
    
    return img

def save_captured_images(images, name, dataset_dir="images dataset"):
    """Save captured images for training"""
    person_dir = os.path.join(dataset_dir, name)
    os.makedirs(person_dir, exist_ok=True)
    
    existing_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    count = len(existing_files)
    saved_paths = []

    for img in images:
        img_pil = Image.fromarray(img)
        filename = f"{count:04d}.jpg"
        path = os.path.join(person_dir, filename)
        img_pil.save(path)
        saved_paths.append(path)
        count += 1

    return saved_paths

def analyze_multiple_photos(images, mtcnn, facenet, svm, encoder, device):
    """Analyze multiple photos and show combined results"""
    st.subheader("📊 Multi-Photo Analysis Results")
    
    all_detections = {}
    total_faces = 0
    
    progress_bar = st.progress(0)
    
    for i, image in enumerate(images):
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = predict_faces(frame, mtcnn, facenet, svm, encoder, device)
        
        total_faces += len(results)
        
        for box, name, conf in results:
            if name not in all_detections:
                all_detections[name] = []
            all_detections[name].append(conf)
        
        progress_bar.progress((i + 1) / len(images))
    
    # Show summary
    st.success(f"✅ Analysis complete! Found {total_faces} faces across {len(images)} photos")
    
    if all_detections:
        st.subheader("🎯 Detection Summary")
        
        for name, confidences in all_detections.items():
            avg_conf = sum(confidences) / len(confidences)
            max_conf = max(confidences)
            detection_count = len(confidences)
            
            if name == "Unknown":
                st.markdown(f'''
                <div class="unknown-box">
                    👤 <strong>{name}</strong><br>
                    📊 Detections: {detection_count}<br>
                    📈 Average confidence: N/A
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="prediction-box">
                    ✅ <strong>{name}</strong><br>
                    📊 Detections: {detection_count}<br>
                    📈 Average confidence: {avg_conf:.1f}%<br>
                    🔝 Best confidence: {max_conf:.1f}%
                </div>
                ''', unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">👤 Face Recognition System</h1>', unsafe_allow_html=True)
    
    # Camera troubleshooting section
    if st.sidebar.checkbox("🛠️ Camera Issues?", help="Check this if you're having camera problems"):
        st.sidebar.markdown("""
        **Quick Fixes:**
        1. **Refresh page** (F5)
        2. **Allow camera** in browser
        3. **Try Chrome browser**
        4. **Use file upload** instead
        5. **Check HTTPS connection**
        """)
        
        if st.sidebar.button("🔄 Force Page Refresh"):
            st.rerun()
    
    # Initialize models
    with st.spinner("🔄 Loading models..."):
        mtcnn, facenet, device = setup_models()
        svm, encoder = load_trained_model()
    
    if svm is None or encoder is None:
        st.error("❌ Cannot proceed without trained models. Please ensure the model files are available.")
        st.info("""
        **Required files:**
        - `svm_model.pkl`
        - `label_encoder.pkl`
        
        These files should be in the same directory as this script.
        """)
        st.stop()
    
    st.success("✅ Models loaded successfully!")
    st.info(f"🖥️ Using device: {device}")
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    mode = st.sidebar.selectbox(
        "Choose Mode:",
        ["📷 Webcam Recognition", "🖼️ Image Upload", "📁 Process Folder", "➕ Register New Face"]
    )
    
    if mode == "📷 Webcam Recognition":
        webcam_recognition(mtcnn, facenet, svm, encoder, device)
    elif mode == "🖼️ Image Upload":
        image_upload_recognition(mtcnn, facenet, svm, encoder, device)
    elif mode == "📁 Process Folder":
        folder_processing(mtcnn, facenet, svm, encoder, device)
    elif mode == "➕ Register New Face":
        face_registration(mtcnn, facenet, device)

def webcam_recognition(mtcnn, facenet, svm, encoder, device):
    st.markdown('<h2 class="section-header">📷 Webcam Face Recognition</h2>', unsafe_allow_html=True)
    
    # Explanation about live vs static
    st.warning("""
    ⚠️ **Important Note about Live Recognition**: 
    
    Unlike the original tkinter app, **Streamlit cannot provide true live webcam streaming** due to browser security limitations. 
    However, we provide several alternatives:
    """)
    
    # Offer different modes
    recognition_mode = st.radio(
        "🎯 Choose Recognition Mode:",
        [
            "📸 Quick Photo Recognition (Recommended)",
            "🔄 Rapid Capture Mode",
            "📱 Continuous Monitoring"
        ],
        help="Select the mode that best fits your needs"
    )
    
    if recognition_mode == "📸 Quick Photo Recognition (Recommended)":
        quick_photo_mode(mtcnn, facenet, svm, encoder, device)
    elif recognition_mode == "🔄 Rapid Capture Mode":
        rapid_capture_mode(mtcnn, facenet, svm, encoder, device)
    else:
        continuous_monitoring_mode(mtcnn, facenet, svm, encoder, device)

def quick_photo_mode(mtcnn, facenet, svm, encoder, device):
    """Single photo capture and analysis"""
    st.subheader("📸 Quick Photo Recognition")
    
    # Camera troubleshooting section
    st.info("🔧 **Camera Status Check**: The browser says camera is in use but you can't see it? Try these solutions:")
    
    # Check browser info
    st.markdown("""
    <script>
    const userAgent = navigator.userAgent;
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent);
    const browserInfo = userAgent.includes('Chrome') ? '✅ Chrome' : 
                       userAgent.includes('Firefox') ? '⚠️ Firefox' : 
                       userAgent.includes('Safari') ? '❌ Safari' : '❓ Unknown';
    
    document.addEventListener('DOMContentLoaded', function() {
        console.log('Browser:', browserInfo, 'Mobile:', isMobile);
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Browser-specific troubleshooting
    st.markdown("""
    **Common Solutions:**
    - 🔄 **Refresh the page** completely (Ctrl+F5)
    - 🎯 **Click directly on the camera button** (not just the area around it)
    - ⏰ **Wait 5-10 seconds** after camera permission is granted
    - 🔍 **Look for a small camera widget** - it might be loading
    - � **Try on mobile** if on desktop (sometimes works better)
    """)
    
    # Alternative approaches
    st.subheader("🎯 Method 1: File Upload (Always Works)")
    uploaded_file = st.file_uploader(
        "Upload a photo - this always works and gives same results",
        type=['png', 'jpg', 'jpeg'],
        help="This is the most reliable method"
    )
    
    if uploaded_file is not None:
        # Process uploaded image
        image = Image.open(uploaded_file)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        with st.spinner("🔍 Analyzing uploaded photo..."):
            results = predict_faces(frame, mtcnn, facenet, svm, encoder, device)
        
        if results:
            img_with_predictions = draw_predictions(image, results)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📸 Uploaded Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("🎯 Recognition Results")
                st.image(img_with_predictions, use_column_width=True)
            
            # Results summary
            st.subheader("📊 Recognition Summary")
            for i, (box, name, conf) in enumerate(results):
                if name == "Unknown":
                    st.error(f"👤 **Face {i+1}**: Unknown Person")
                else:
                    st.success(f"✅ **Face {i+1}**: {name} ({conf:.1f}% confidence)")
        else:
            st.warning("😔 No faces detected in uploaded image.")
    
    st.markdown("---")
    st.subheader("📷 Method 2: Camera Capture")
    
    # Multiple camera attempts
    st.info("👀 **Look carefully below** - the camera widget might be small or take time to load")
    
    # Camera diagnostic
    if st.button("🔍 Test Camera Access"):
        st.info("� Testing camera... Look for camera widget below")
        st.warning("⚠️ If you see this message but no camera, try refreshing the page")
    
    # Simple camera first
    st.markdown("**Standard Camera:**")
    camera_input = st.camera_input(
        "📷 Camera should appear here - look carefully",
        help="If you don't see a camera widget, scroll up and try the file upload instead"
    )
    
    # Process camera input
    if camera_input is not None:
        # Convert to numpy array
        image = Image.open(camera_input)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        with st.spinner("🔍 Analyzing faces..."):
            results = predict_faces(frame, mtcnn, facenet, svm, encoder, device)
        
        if results:
            img_with_predictions = draw_predictions(image, results)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📸 Captured Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("🎯 Recognition Results")
                st.image(img_with_predictions, use_column_width=True)
            
            # Results summary
            st.subheader("📊 Recognition Summary")
            for i, (box, name, conf) in enumerate(results):
                if name == "Unknown":
                    st.error(f"👤 **Face {i+1}**: Unknown Person")
                else:
                    st.success(f"✅ **Face {i+1}**: {name} ({conf:.1f}% confidence)")
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Results", key="save_quick"):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"recognition_{timestamp}.jpg"
                    img_with_predictions.save(filename)
                    st.success(f"✅ Saved as {filename}")
            
            with col2:
                if st.button("🔄 Take Another Photo", key="retake_quick"):
                    st.rerun()
        else:
            st.warning("😔 No faces detected. Please try again with better lighting and positioning.")
    
    # Alternative camera methods
    st.markdown("---")
    st.markdown("**Alternative Camera Methods:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Method A: Reset Camera**")
        if st.button("🔄 Reset Camera Widget"):
            st.rerun()
    
    with col2:
        st.markdown("**Method B: Force New Camera**")
        unique_key = f"camera_new_{random.randint(1000, 9999)}"
        camera_input_alt = st.camera_input(
            "Alternative camera",
            key=unique_key,
            help="Different camera instance"
        )
        
        if camera_input_alt is not None:
            st.success("✅ Alternative camera worked!")
            # Same processing as above...
    
    # Final troubleshooting
    with st.expander("🛠️ Still No Camera? Complete Troubleshooting", expanded=False):
        st.markdown("""
        **If camera still doesn't appear:**
        
        1. **Check URL**: Make sure you're on the deployed app (not localhost)
        2. **Browser Console**: Press F12 → Console tab → look for errors
        3. **Camera Permissions**: 
           - Chrome: Click 🔒 icon → Camera → Allow
           - Edge: Click 🔒 icon → Permissions → Camera → Allow
        4. **Different Browser**: Try Chrome, Edge, or Firefox
        5. **Disable Extensions**: Ad blockers might interfere
        6. **Mobile Device**: Sometimes works better on phones/tablets
        7. **Incognito Mode**: Try in private/incognito window
        8. **Clear Cache**: Clear browser cache and try again
        
        **Known Issues:**
        - Some corporate networks block camera access
        - VPNs might interfere with camera permissions
        - Antivirus software can block camera access
        - Multiple apps using camera can cause conflicts
        
        **Alternative Solutions:**
        - Use the file upload method (works 100% of the time)
        - Take photos with your phone and upload them
        - Use a different device or network
        """)
    
    # Success indicators
    if camera_input is not None:
        st.balloons()
        st.success("🎉 Camera is working! You successfully captured a photo!")

def rapid_capture_mode(mtcnn, facenet, svm, encoder, device):
    """Multiple rapid captures for quasi-live experience"""
    st.subheader("🔄 Rapid Capture Mode")
    st.info("Take multiple quick photos in sequence for a near-live experience")
    
    # Initialize session state
    if 'rapid_captures' not in st.session_state:
        st.session_state.rapid_captures = []
    if 'auto_capture' not in st.session_state:
        st.session_state.auto_capture = False
    
    # Controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🎬 Start Auto-Capture"):
            st.session_state.auto_capture = True
            st.success("Auto-capture mode activated!")
    
    with col2:
        if st.button("⏹️ Stop Auto-Capture"):
            st.session_state.auto_capture = False
            st.info("Auto-capture stopped")
    
    with col3:
        if st.button("🗑️ Clear All"):
            st.session_state.rapid_captures = []
            st.success("All captures cleared!")
    
    with col4:
        capture_count = len(st.session_state.rapid_captures)
        st.metric("� Captures", capture_count)
    
    # Camera input
    camera_key = f"rapid_camera_{len(st.session_state.rapid_captures)}"
    camera_input = st.camera_input(
        "📷 Quick Capture (take multiple photos rapidly)",
        key=camera_key,
        help="Take photos quickly one after another for a live-like experience"
    )
    
    if camera_input is not None:
        image = Image.open(camera_input)
        st.session_state.rapid_captures.append(image)
        
        # Auto-analyze the latest capture
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = predict_faces(frame, mtcnn, facenet, svm, encoder, device)
        
        # Show latest result
        st.subheader("🎯 Latest Recognition Result")
        if results:
            img_with_predictions = draw_predictions(image, results)
            st.image(img_with_predictions, caption=f"Capture #{len(st.session_state.rapid_captures)}", use_column_width=True)
            
            for i, (box, name, conf) in enumerate(results):
                if name == "Unknown":
                    st.error(f"👤 Face {i+1}: Unknown")
                else:
                    st.success(f"✅ Face {i+1}: {name} ({conf:.1f}%)")
        else:
            st.warning("No faces detected in latest capture")
    
    # Show capture history
    if st.session_state.rapid_captures:
        st.subheader("📈 Capture History")
        if st.button("� Analyze All Captures"):
            analyze_capture_history(st.session_state.rapid_captures, mtcnn, facenet, svm, encoder, device)

def continuous_monitoring_mode(mtcnn, facenet, svm, encoder, device):
    """Simulated continuous monitoring with periodic captures"""
    st.subheader("📱 Continuous Monitoring Mode")
    st.info("Simulates continuous monitoring by analyzing periodic captures")
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        monitoring_active = st.checkbox("🔴 Activate Monitoring", help="Enable continuous monitoring mode")
    with col2:
        confidence_threshold = st.slider("🎯 Confidence Threshold", 0, 100, 70, help="Minimum confidence for positive identification")
    
    if monitoring_active:
        st.success("🔴 **MONITORING ACTIVE** - System is watching for faces")
        
        # Instructions for manual captures during monitoring
        st.info("""
        **Monitoring Instructions:**
        1. Keep this page open
        2. Take photos periodically using the camera below
        3. The system will track all detections
        4. View the monitoring log for a history of detections
        """)
        
        # Initialize monitoring session
        if 'monitoring_log' not in st.session_state:
            st.session_state.monitoring_log = []
        if 'monitoring_stats' not in st.session_state:
            st.session_state.monitoring_stats = {}
        
        # Camera for monitoring
        monitor_camera = st.camera_input(
            "📹 Monitoring Camera (take photos to simulate monitoring)",
            key="monitoring_camera",
            help="Each photo represents a monitoring checkpoint"
        )
        
        if monitor_camera is not None:
            timestamp = time.strftime("%H:%M:%S")
            image = Image.open(monitor_camera)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            results = predict_faces(frame, mtcnn, facenet, svm, encoder, device)
            
            # Log the detection
            detection_entry = {
                'timestamp': timestamp,
                'faces_detected': len(results),
                'results': results,
                'image': image
            }
            st.session_state.monitoring_log.append(detection_entry)
            
            # Update stats
            for box, name, conf in results:
                if conf >= confidence_threshold and name != "Unknown":
                    if name not in st.session_state.monitoring_stats:
                        st.session_state.monitoring_stats[name] = 0
                    st.session_state.monitoring_stats[name] += 1
            
            # Show current detection
            if results:
                st.success(f"🕒 **{timestamp}**: Detected {len(results)} face(s)")
                for i, (box, name, conf) in enumerate(results):
                    if name != "Unknown" and conf >= confidence_threshold:
                        st.info(f"✅ **{name}** detected ({conf:.1f}% confidence)")
                    else:
                        st.warning(f"❓ Unknown person or low confidence ({conf:.1f}%)")
            else:
                st.info(f"� **{timestamp}**: No faces detected")
        
        # Show monitoring dashboard
        if st.session_state.monitoring_log:
            st.subheader("📊 Monitoring Dashboard")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Checks", len(st.session_state.monitoring_log))
            with col2:
                total_faces = sum(entry['faces_detected'] for entry in st.session_state.monitoring_log)
                st.metric("👥 Total Faces", total_faces)
            with col3:
                st.metric("👤 Unique People", len(st.session_state.monitoring_stats))
            
            # Recent activity
            st.subheader("🕒 Recent Activity")
            for entry in st.session_state.monitoring_log[-5:]:  # Show last 5 entries
                st.text(f"{entry['timestamp']}: {entry['faces_detected']} face(s) detected")
            
            if st.button("📋 View Full Log"):
                show_monitoring_log()
    else:
        st.info("👁️ Monitoring is inactive. Check the box above to start monitoring.")

def analyze_capture_history(images, mtcnn, facenet, svm, encoder, device):
    """Analyze the history of rapid captures"""
    st.subheader("� Capture History Analysis")
    
    face_timeline = []
    
    for i, image in enumerate(images):
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = predict_faces(frame, mtcnn, facenet, svm, encoder, device)
        face_timeline.append((i+1, len(results), results))
    
    # Show timeline
    st.subheader("📊 Detection Timeline")
    for capture_num, face_count, results in face_timeline:
        if face_count > 0:
            names = [name for _, name, conf in results if conf > 40]
            unique_names = list(set(names))
            if unique_names:
                st.success(f"📸 Capture {capture_num}: {', '.join(unique_names)}")
            else:
                st.warning(f"📸 Capture {capture_num}: Unknown faces")
        else:
            st.info(f"📸 Capture {capture_num}: No faces")

def show_monitoring_log():
    """Display the full monitoring log"""
    st.subheader("📋 Full Monitoring Log")
    
    for i, entry in enumerate(reversed(st.session_state.monitoring_log)):
        with st.expander(f"🕒 {entry['timestamp']} - {entry['faces_detected']} face(s)"):
            if entry['results']:
                img_with_pred = draw_predictions(entry['image'], entry['results'])
                st.image(img_with_pred, use_column_width=True)
                for j, (box, name, conf) in enumerate(entry['results']):
                    st.text(f"Face {j+1}: {name} ({conf:.1f}%)")
            else:
                st.image(entry['image'], use_column_width=True)
                st.text("No faces detected")

def image_upload_recognition(mtcnn, facenet, svm, encoder, device):
    st.markdown('<h2 class="section-header">🖼️ Image Upload Recognition</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image containing faces for recognition"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize if too large
        max_size = 800
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size))
            frame = cv2.resize(frame, (image.size[0], image.size[1]))
        
        with st.spinner("🔍 Analyzing faces..."):
            results = predict_faces(frame, mtcnn, facenet, svm, encoder, device)
        
        if results:
            # Draw predictions
            img_with_predictions = draw_predictions(image, results)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📸 Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("🎯 Recognition Results")
                st.image(img_with_predictions, use_column_width=True)
            
            # Display predictions
            st.subheader("📊 Detection Details")
            for i, (box, name, conf) in enumerate(results):
                if name == "Unknown":
                    st.markdown(f'<div class="unknown-box">👤 Face {i+1}: {name}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box">✅ Face {i+1}: {name} ({conf:.1f}% confidence)</div>', unsafe_allow_html=True)
        else:
            st.warning("😔 No faces detected in the image. Please try again with a clearer image.")

def folder_processing(mtcnn, facenet, svm, encoder, device):
    st.markdown('<h2 class="section-header">📁 Process Multiple Images</h2>', unsafe_allow_html=True)
    
    st.info("📋 Upload multiple images to process them in batch")
    
    # Processing mode selection
    processing_mode = st.radio(
        "🎯 Choose processing mode:",
        ["🔍 Face Recognition", "👥 Face Grouping (DBSCAN Clustering)"],
        help="Recognition: Identify known faces | Grouping: Cluster unknown faces by similarity"
    )
    
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload multiple images for batch processing"
    )
    
    if uploaded_files:
        if processing_mode == "🔍 Face Recognition":
            process_recognition_batch(uploaded_files, mtcnn, facenet, svm, encoder, device)
        else:
            process_face_grouping(uploaded_files, mtcnn, facenet, device)

def process_recognition_batch(uploaded_files, mtcnn, facenet, svm, encoder, device):
    """Process uploaded files for face recognition"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Load and process image
        image = Image.open(uploaded_file)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize if too large
        max_size = 600
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size))
            frame = cv2.resize(frame, (image.size[0], image.size[1]))
        
        results = predict_faces(frame, mtcnn, facenet, svm, encoder, device)
        
        if results:
            img_with_predictions = draw_predictions(image, results)
            all_results.append((uploaded_file.name, image, img_with_predictions, results))
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("✅ Processing complete!")
    
    # Display results
    if all_results:
        st.subheader("📊 Batch Processing Results")
        
        for filename, original, predicted, results in all_results:
            st.markdown(f"### 📄 {filename}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(original, caption="Original", use_column_width=True)
            with col2:
                st.image(predicted, caption="With Predictions", use_column_width=True)
            
            # Show detection details
            for j, (box, name, conf) in enumerate(results):
                if name == "Unknown":
                    st.markdown(f'<div class="unknown-box">👤 Face {j+1}: {name}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box">✅ Face {j+1}: {name} ({conf:.1f}% confidence)</div>', unsafe_allow_html=True)
            
            st.markdown("---")
    else:
        st.warning("😔 No faces detected in any of the uploaded images.")

def process_face_grouping(uploaded_files, mtcnn, facenet, device):
    """Process uploaded files for face grouping using DBSCAN clustering"""
    st.subheader("👥 Face Grouping Analysis")
    
    # Clustering parameters
    col1, col2 = st.columns(2)
    with col1:
        eps = st.slider("🎯 Clustering sensitivity (eps)", 0.1, 1.0, 0.5, 0.05, 
                       help="Lower = stricter grouping, Higher = looser grouping")
    with col2:
        min_samples = st.slider("👥 Minimum group size", 1, 5, 2,
                               help="Minimum faces needed to form a group")
    
    if st.button("🔄 Start Clustering Analysis"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract all face embeddings
        all_embeddings = []
        face_info = []  # [(filename, face_index, box, embedding, image)]
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Extracting faces from {uploaded_file.name}...")
            
            image = Image.open(uploaded_file)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Resize if too large
            max_size = 600
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size))
                frame = cv2.resize(frame, (image.size[0], image.size[1]))
            
            faces = get_faces_and_embeddings(frame, mtcnn, facenet, device, min_conf=0.90)
            
            for j, (box, embedding, conf) in enumerate(faces):
                all_embeddings.append(embedding)
                face_info.append((uploaded_file.name, j, box, embedding, image))
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if len(all_embeddings) == 0:
            st.warning("😔 No faces detected in any images.")
            return
        
        status_text.text("🔍 Performing clustering analysis...")
        
        # Perform DBSCAN clustering
        from sklearn.cluster import DBSCAN
        
        # Normalize embeddings
        normalized_embeddings = Normalizer(norm='l2').transform(all_embeddings)
        
        # Apply DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(normalized_embeddings)
        
        # Group faces by cluster
        clusters = defaultdict(list)
        noise_faces = []
        
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise/outlier
                noise_faces.append(face_info[i])
            else:
                clusters[label].append(face_info[i])
        
        status_text.text("✅ Clustering complete!")
        
        # Display results
        st.subheader(f"📊 Clustering Results")
        st.info(f"Found {len(clusters)} groups and {len(noise_faces)} unique faces")
        
        # Show clusters
        for cluster_id, faces in clusters.items():
            with st.expander(f"👥 Group {cluster_id + 1} ({len(faces)} faces)", expanded=True):
                cols = st.columns(min(4, len(faces)))
                
                for i, (filename, face_idx, box, embedding, image) in enumerate(faces):
                    with cols[i % 4]:
                        # Crop face from image
                        face_crop = image.crop(box)
                        st.image(face_crop, caption=f"{filename}\nFace {face_idx + 1}", use_column_width=True)
                
                # Option to save group
                group_name = st.text_input(f"💾 Save Group {cluster_id + 1} as:", 
                                         placeholder="Enter person name", 
                                         key=f"group_name_{cluster_id}")
                
                if group_name and st.button(f"Save Group {cluster_id + 1}", key=f"save_group_{cluster_id}"):
                    # Save all faces in this group
                    group_images = []
                    for filename, face_idx, box, embedding, image in faces:
                        face_crop = image.crop(box)
                        group_images.append(cv2.cvtColor(np.array(face_crop), cv2.COLOR_RGB2BGR))
                    
                    saved_paths = save_captured_images(group_images, group_name, "images dataset")
                    st.success(f"✅ Saved {len(saved_paths)} faces as '{group_name}'")
        
        # Show noise/unique faces
        if noise_faces:
            with st.expander(f"🔍 Unique Faces ({len(noise_faces)} faces)", expanded=False):
                cols = st.columns(min(4, len(noise_faces)))
                
                for i, (filename, face_idx, box, embedding, image) in enumerate(noise_faces):
                    with cols[i % 4]:
                        face_crop = image.crop(box)
                        st.image(face_crop, caption=f"{filename}\nFace {face_idx + 1}", use_column_width=True)
                        
                        # Individual save option
                        if st.button(f"💾 Save", key=f"save_unique_{i}"):
                            person_name = st.text_input(f"Name for face {i+1}:", key=f"unique_name_{i}")
                            if person_name:
                                face_array = cv2.cvtColor(np.array(face_crop), cv2.COLOR_RGB2BGR)
                                save_single_image(face_array, person_name, "images dataset")
                                st.success(f"✅ Saved as '{person_name}'")

def face_registration(mtcnn, facenet, device):
    st.markdown('<h2 class="section-header">➕ Register New Face</h2>', unsafe_allow_html=True)
    
    st.info("📝 Register a new person or add photos to existing person")
    
    # Create dataset directory
    dataset_dir = "images dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Get existing people in dataset
    existing_people = []
    if os.path.exists(dataset_dir):
        existing_people = [name for name in os.listdir(dataset_dir) 
                          if os.path.isdir(os.path.join(dataset_dir, name))]
    
    # Person selection/creation
    st.subheader("👤 Select or Create Person")
    
    person_mode = st.radio(
        "Choose option:",
        ["➕ Create New Person", "📝 Add to Existing Person"],
        help="Choose whether to create a new person or add photos to existing person"
    )
    
    if person_mode == "➕ Create New Person":
        person_name = st.text_input("👤 Enter new person's name:", placeholder="e.g., John Doe")
        
        if person_name:
            if person_name in existing_people:
                st.error(f"❌ Person '{person_name}' already exists! Choose 'Add to Existing Person' option above.")
                person_name = None
    else:
        if existing_people:
            person_name = st.selectbox(
                "👥 Select existing person:",
                [""] + existing_people,
                help="Choose an existing person to add more photos"
            )
            if person_name == "":
                person_name = None
        else:
            st.warning("📭 No existing people found. Please create a new person first.")
            person_name = None
    
    if person_name:
        person_dir = os.path.join(dataset_dir, person_name)
        
        # Show existing info
        if os.path.exists(person_dir):
            existing_count = len([f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            st.info(f"📊 Current photos for '{person_name}': {existing_count}")
            
            # Show some existing photos
            if existing_count > 0:
                with st.expander(f"👁️ View existing photos for {person_name}", expanded=False):
                    existing_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:6]
                    cols = st.columns(3)
                    for i, filename in enumerate(existing_files):
                        with cols[i % 3]:
                            img_path = os.path.join(person_dir, filename)
                            img = Image.open(img_path)
                            st.image(img, caption=filename, use_column_width=True)
        
        st.subheader(f"📷 Add photos for {person_name}")
        
        # Duplicate detection settings
        st.subheader("🔍 Duplicate Detection Settings")
        col1, col2 = st.columns(2)
        with col1:
            enable_duplicate_check = st.checkbox("🛡️ Enable duplicate detection", value=True, 
                                                help="Check if face already exists in dataset")
        with col2:
            similarity_threshold = st.slider("🎯 Similarity threshold", 0.1, 1.0, 0.8, 0.1,
                                            help="Lower = more strict duplicate detection")
        
        # Option 1: Camera capture
        st.subheader("📱 Option 1: Camera Capture")
        st.info("💡 Take multiple photos for better training accuracy!")
        
        camera_input = st.camera_input(
            "Take photos (capture different angles and expressions)",
            help="Each photo will be validated and checked for duplicates before adding"
        )
        
        if camera_input is not None:
            image = Image.open(camera_input)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect faces to ensure there's exactly one face
            faces = get_faces_and_embeddings(frame, mtcnn, facenet, device, min_conf=0.90)
            
            if len(faces) == 1:
                st.success("✅ Good photo! One face detected clearly.")
                
                # Check for duplicates if enabled
                is_duplicate = False
                if enable_duplicate_check and os.path.exists(person_dir):
                    is_duplicate = check_face_duplicate(faces[0][1], person_dir, mtcnn, facenet, device, similarity_threshold)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Captured Photo", use_column_width=True)
                
                with col2:
                    if is_duplicate:
                        st.warning("⚠️ Similar face already exists in dataset!")
                        if st.button("� Add Anyway", key="add_duplicate_camera"):
                            save_single_image(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), person_name, dataset_dir)
                            st.success("✅ Photo added despite similarity!")
                    else:
                        if st.button("� Add to Dataset", key="save_camera_validated"):
                            save_single_image(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), person_name, dataset_dir)
                            st.success("✅ Photo added successfully!")
                            st.info("� You may need to retrain the model to recognize new photos.")
                    
            elif len(faces) == 0:
                st.error("❌ No face detected. Please take a clearer photo.")
                st.info("💡 Tips: Ensure good lighting, face the camera directly, remove glasses if needed.")
            else:
                st.error(f"❌ Multiple faces detected ({len(faces)}). Please ensure only one person is in the photo.")
        
        # Option 2: File upload
        st.subheader("📁 Option 2: Upload Photos")
        uploaded_files = st.file_uploader(
            "Upload photos",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload multiple clear photos of the person for better accuracy"
        )
        
        if uploaded_files:
            valid_images = []
            duplicate_images = []
            
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                faces = get_faces_and_embeddings(frame, mtcnn, facenet, device, min_conf=0.90)
                
                if len(faces) == 1:
                    # Check for duplicates
                    is_duplicate = False
                    if enable_duplicate_check and os.path.exists(person_dir):
                        is_duplicate = check_face_duplicate(faces[0][1], person_dir, mtcnn, facenet, device, similarity_threshold)
                    
                    if is_duplicate:
                        duplicate_images.append((uploaded_file.name, frame, image))
                        st.warning(f"⚠️ {uploaded_file.name}: Similar face already exists")
                    else:
                        valid_images.append((uploaded_file.name, frame, image))
                        st.success(f"✅ {uploaded_file.name}: Valid unique photo")
                elif len(faces) == 0:
                    st.error(f"❌ {uploaded_file.name}: No face detected")
                else:
                    st.error(f"❌ {uploaded_file.name}: Multiple faces detected")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Show results
            if valid_images:
                st.subheader("✅ Valid Unique Photos")
                cols = st.columns(min(3, len(valid_images)))
                for i, (filename, frame, image) in enumerate(valid_images):
                    with cols[i % 3]:
                        st.image(image, caption=filename, use_column_width=True)
                
                if st.button(f"💾 Save {len(valid_images)} unique photos", key="save_valid"):
                    frames = [frame for _, frame, _ in valid_images]
                    saved_paths = save_captured_images(frames, person_name, dataset_dir)
                    total_photos = len([f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                    st.success(f"✅ {len(saved_paths)} photos saved! Total for {person_name}: {total_photos}")
                    st.info("💡 You may need to retrain the model to recognize new photos.")
            
            if duplicate_images:
                st.subheader("⚠️ Potential Duplicate Photos")
                cols = st.columns(min(3, len(duplicate_images)))
                for i, (filename, frame, image) in enumerate(duplicate_images):
                    with cols[i % 3]:
                        st.image(image, caption=f"⚠️ {filename}", use_column_width=True)
                
                if st.button(f"🔄 Save {len(duplicate_images)} duplicates anyway", key="save_duplicates"):
                    frames = [frame for _, frame, _ in duplicate_images]
                    saved_paths = save_captured_images(frames, person_name, dataset_dir)
                    st.success(f"✅ {len(saved_paths)} duplicate photos saved anyway!")
        
        # Training recommendation
        if os.path.exists(person_dir):
            photo_count = len([f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            
            st.subheader("📊 Training Recommendations")
            if photo_count < 5:
                st.warning(f"⚠️ Only {photo_count} photos. Recommended: 5-10 photos for better accuracy.")
            elif photo_count < 10:
                st.info(f"✅ {photo_count} photos. Good! Add 1-2 more for optimal accuracy.")
            else:
                st.success(f"🎉 {photo_count} photos. Excellent dataset size!")

def check_face_duplicate(new_embedding, person_dir, mtcnn, facenet, device, threshold=0.8):
    """Check if a face embedding is similar to existing faces in the person's directory"""
    try:
        existing_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for filename in existing_files:
            img_path = os.path.join(person_dir, filename)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            
            faces = get_faces_and_embeddings(frame, mtcnn, facenet, device, min_conf=0.90)
            
            for _, existing_embedding, _ in faces:
                # Calculate cosine similarity
                similarity = cosine_similarity([new_embedding], [existing_embedding])[0][0]
                
                if similarity > threshold:
                    return True
        
        return False
    except Exception as e:
        st.warning(f"⚠️ Duplicate check failed: {e}")
        return False

def save_single_image(image_array, name, dataset_dir="images dataset"):
    """Save a single image to the dataset"""
    person_dir = os.path.join(dataset_dir, name)
    os.makedirs(person_dir, exist_ok=True)
    
    existing_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    count = len(existing_files)
    
    img_pil = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    filename = f"{count:04d}.jpg"
    path = os.path.join(person_dir, filename)
    img_pil.save(path)
    
    return path

if __name__ == "__main__":
    main()
