import streamlit as st

# Simple test to see if the app loads
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide"
)

st.markdown('<h1 style="color: #1E90FF; text-align: center;">👤 Face Recognition System</h1>', unsafe_allow_html=True)

st.success("✅ App is loading successfully!")

# Test if required files exist
import os

files_to_check = ['svm_model.pkl', 'label_encoder.pkl', 'embeddings.npy', 'labels.npy']
for file in files_to_check:
    if os.path.exists(file):
        st.success(f"✅ {file} found")
    else:
        st.error(f"❌ {file} missing")

# Simple navigation test
st.sidebar.title("🧭 Navigation")
mode = st.sidebar.selectbox(
    "Choose Mode:",
    ["📷 Webcam Recognition", "🖼️ Image Upload", "📁 Process Folder", "➕ Register New Face"]
)

st.info(f"Selected mode: {mode}")

# Test camera input
if mode == "📷 Webcam Recognition":
    st.subheader("📷 Webcam Test")
    camera_input = st.camera_input("Test camera")
    if camera_input:
        st.success("✅ Camera is working!")
        st.image(camera_input)

st.info("🎉 Basic app structure is working! If you see this message, the deployment issue should be resolved.")
