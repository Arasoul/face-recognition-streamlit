import logging
import torch
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import Normalizer
import joblib
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import simpledialog, messagebox
import threading
import queue
import os

from sklearn.cluster import DBSCAN
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import math
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_models(image_size=160, margin=20, pretrained='vggface2'):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        mtcnn = MTCNN(image_size=image_size, margin=margin, keep_all=True, device=device)
        facenet = InceptionResnetV1(pretrained=pretrained).eval().to(device)
        return mtcnn, facenet, device
    except Exception as e:
        logging.error(f"Failed to setup models: {e}")
        raise

def load_model(svm_path='svm_model.pkl', encoder_path='label_encoder.pkl'):
    try:
        svm = joblib.load(svm_path)
        encoder = joblib.load(encoder_path)
        logging.info(f"Loaded model from {svm_path} and encoder from {encoder_path}")
        return svm, encoder
    except Exception as e:
        logging.error(f"Failed to load model or encoder: {e}")
        return None, None

def get_faces_and_embeddings(frame, mtcnn, facenet, device, min_conf=0.90):
    try:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, probs = mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            return []

        faces = []
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob >= min_conf:
                try:
                    # Ensure box coordinates are valid
                    box = [max(0, int(coord)) for coord in box]
                    face_tensor = mtcnn.extract(img, [box], save_path=None)
                    if face_tensor is None or len(face_tensor) == 0:
                        continue
                    face = face_tensor[0].unsqueeze(0).to(device)
                    with torch.no_grad():
                        embedding = facenet(face).cpu().numpy()
                    faces.append((box, embedding.flatten(), prob))
                except Exception as e:
                    logging.warning(f"Failed to process face {i}: {e}")
                    continue
        return faces
    except Exception as e:
        logging.error(f"Error in get_faces_and_embeddings: {e}")
        return []

def predict_faces(frame, mtcnn, facenet, svm, encoder, device):
    results = []
    if svm is None or encoder is None:
        return results  # Do nothing if model is untrained

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
        except (ValueError, IndexError) as e:
            logging.warning(f"Prediction error: {e}")
            identity, confidence = 'Unknown', 0.0
        except Exception as e:
            logging.error(f"Unexpected error in prediction: {e}")
            identity, confidence = 'Unknown', 0.0
        results.append((box, identity, confidence))
    return results

def save_captured_images(images, name, dataset_dir="images dataset"):
    if not images or not name:
        raise ValueError("Images list and name cannot be empty")
    
    # Sanitize name to avoid filesystem issues
    name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
    if not name:
        raise ValueError("Invalid name after sanitization")

    person_dir = os.path.join(dataset_dir, name)
    os.makedirs(person_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    count = len(existing_files)
    saved_paths = []

    for img in images:
        img_pil = Image.fromarray(img)
        filename = f"{count:04d}.jpg"
        path = os.path.join(person_dir, filename)
        img_pil.save(path, quality=95)  # Add quality parameter
        saved_paths.append(path)
        count += 1

    return saved_paths

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Face Recognition System")
        self.root.configure(bg='#2c3e50')
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        
        # Setup simple modern styling
        self.setup_simple_styles()
        
        self.mtcnn, self.facenet, self.device = setup_models()
        self.svm, self.encoder = load_model()
        if self.svm is None:
            logging.warning("No trained model found. You can start by registering faces.")
            messagebox.showinfo("No Model", "No trained model found. Please register faces first to start recognition.")
        
        # Camera management
        self.frame_count = 0
        self.running = False
        self.running_registration = False
        self.cap = None
        self.queue = queue.Queue()
        self.thread = None
        self.camera_mode = None
        
        # Initialize basic components
        self.scroll_frame = tk.Frame(self.root, bg='#2c3e50')
        self.thumbnails = []
        
        # Initialize scroll canvas components for folder processing
        self.canvas_frame = None
        self.scroll_canvas = None
        
        # Initialize AI status labels dictionary
        self.ai_status_labels = {}
        
        # Initialize scanning effect
        self.scanning_angle = 0
        
        # Create simple GUI
        self.create_simple_gui()
        
        # Initialize camera after GUI is ready
        self._initialize_camera()

    def setup_simple_styles(self):
        """Setup simple, fast styling"""
        style = ttk.Style()
        
        # Use a theme that supports better text visibility
        try:
            style.theme_use('clam')  # Better theme for custom styling
        except:
            pass
        
        # Simple modern button style
        style.configure('Modern.TButton',
                       background='#3498db',
                       foreground='white',
                       borderwidth=1,
                       focuscolor='none',
                       relief='flat',
                       padding=(15, 8),
                       font=('Segoe UI', 10, 'bold'))
        
        style.map('Modern.TButton',
                 background=[('active', '#2980b9'),
                           ('pressed', '#21618c'),
                           ('!active', '#3498db')],
                 foreground=[('active', 'white'),
                           ('pressed', 'white'),
                           ('!active', 'white')])
        
        # Accent button
        style.configure('Accent.TButton',
                       background='#e74c3c',
                       foreground='white',
                       borderwidth=1,
                       focuscolor='none',
                       relief='flat',
                       padding=(12, 6),
                       font=('Segoe UI', 9, 'bold'))
        
        style.map('Accent.TButton',
                 background=[('active', '#c0392b'),
                           ('pressed', '#a93226'),
                           ('!active', '#e74c3c')],
                 foreground=[('active', 'white'),
                           ('pressed', 'white'),
                           ('!active', 'white')])

    def create_simple_gui(self):
        """Create a simple, fast GUI"""
        # Header
        header_frame = tk.Frame(self.root, bg='#34495e', height=80)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, 
                              text="🎯 Face Recognition System", 
                              font=("Segoe UI", 20, "bold"), 
                              fg="white", 
                              bg="#34495e")
        title_label.pack(expand=True)
        
        # Main content area
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        control_frame.pack(fill="x", pady=(0, 10))
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg='#34495e')
        button_frame.pack(pady=15)
        
        # TTK Buttons (current implementation)
        self.btn_cam = ttk.Button(button_frame, 
                                 text="📹 Start Camera", 
                                 command=self.start_camera,
                                 style='Modern.TButton')
        self.btn_cam.pack(side="left", padx=5)

        self.btn_upload = ttk.Button(button_frame, 
                                    text="🖼️ Upload Image", 
                                    command=self.upload_image,
                                    style='Modern.TButton')
        self.btn_upload.pack(side="left", padx=5)

        self.btn_folder = ttk.Button(button_frame, 
                                    text="📁 Process Folder", 
                                    command=self.process_folder,
                                    style='Modern.TButton')
        self.btn_folder.pack(side="left", padx=5)

        self.btn_register = ttk.Button(button_frame, 
                                      text="➕ Register Face", 
                                      command=self.register_face,
                                      style='Accent.TButton')
        self.btn_register.pack(side="left", padx=5)
        
        # Display area
        display_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        display_frame.pack(fill="both", expand=True)
        
        # Camera display
        self.label_camera = tk.Label(display_frame, 
                                    bg='#2c3e50', 
                                    relief='sunken', 
                                    bd=2,
                                    text="📷 Camera feed will appear here\nClick 'Start Camera' to begin",
                                    font=("Segoe UI", 14),
                                    fg="#bdc3c7")
        self.label_camera.pack(expand=True, fill="both", padx=10, pady=10)

        # Results area
        self.label_prediction = tk.Label(display_frame, 
                                        text="", 
                                        font=("Segoe UI", 12), 
                                        fg="#3498db", 
                                        bg="#34495e",
                                        wraplength=800)
        self.label_prediction.pack(pady=10)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(fill="x", side="bottom")
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, 
                                    text="🟢 Ready", 
                                    font=("Segoe UI", 9), 
                                    fg="white", 
                                    bg="#34495e")
        self.status_label.pack(side="left", padx=15, pady=5)

    def create_cyberpunk_header(self, parent):
        """Create an animated cyberpunk-style header"""
        header_frame = tk.Frame(parent, bg='#0a0a0a', height=120)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        # Create canvas for animated background
        header_canvas = tk.Canvas(header_frame, bg='#0a0a0a', height=120, highlightthickness=0)
        header_canvas.pack(fill="both", expand=True)
        
        # Draw animated circuit patterns (simplified for text)
        self.draw_circuit_pattern(header_canvas)
        
        # Main title with glow effect
        title_frame = tk.Frame(header_canvas, bg='#0a0a0a')
        header_canvas.create_window(header_canvas.winfo_reqwidth()//2, 30, window=title_frame)
        
        self.main_title = tk.Label(title_frame,
                                  text="🚀 NEURAL FACE RECOGNITION MATRIX",
                                  font=("Consolas", 26, "bold"),
                                  fg="#00ffff",
                                  bg="#0a0a0a")
        self.main_title.pack()
        
        subtitle = tk.Label(title_frame,
                           text="◢ AI-POWERED BIOMETRIC IDENTIFICATION SYSTEM ◣",
                           font=("Consolas", 12),
                           fg="#ff00aa",
                           bg="#0a0a0a")
        subtitle.pack(pady=(5, 0))
        
        # Version and status indicators
        indicators_frame = tk.Frame(header_canvas, bg='#0a0a0a')
        header_canvas.create_window(header_canvas.winfo_reqwidth()//2, 90, window=indicators_frame)
        
        tk.Label(indicators_frame, text="█ STATUS: ACTIVE", font=("Consolas", 9), fg="#00ff00", bg="#0a0a0a").pack(side="left", padx=10)
        tk.Label(indicators_frame, text="█ VERSION: 3.0", font=("Consolas", 9), fg="#ffaa00", bg="#0a0a0a").pack(side="left", padx=10)
        tk.Label(indicators_frame, text="█ NEURAL NET: ONLINE", font=("Consolas", 9), fg="#00aaff", bg="#0a0a0a").pack(side="left", padx=10)

    def draw_circuit_pattern(self, canvas):
        """Draw animated circuit-like patterns"""
        canvas.create_line(50, 10, 200, 10, fill="#333333", width=2)
        canvas.create_line(50, 110, 200, 110, fill="#333333", width=2)
        canvas.create_oval(45, 5, 55, 15, fill="#00ffff", outline="#00ffff")
        canvas.create_oval(195, 5, 205, 15, fill="#ff00aa", outline="#ff00aa")

    def create_control_matrix(self, parent):
        """Create cyberpunk-style control panel"""
        control_frame = tk.Frame(parent, bg='#111111', relief='solid', bd=2)
        control_frame.pack(side="left", fill="y", padx=(0, 15), pady=10)
        
        # Control panel header
        control_header = tk.Frame(control_frame, bg='#1a1a1a', height=50)
        control_header.pack(fill="x", padx=2, pady=2)
        control_header.pack_propagate(False)
        
        tk.Label(control_header, text="⚡ CONTROL MATRIX ⚡", 
                font=("Consolas", 14, "bold"), fg="#00ffff", bg="#1a1a1a").pack(expand=True)
        
        # Button container
        button_container = tk.Frame(control_frame, bg='#111111')
        button_container.pack(fill="both", expand=True, padx=15, pady=20)
        
        # Main control buttons with cyberpunk styling
        buttons_config = [
            ("🎯 ACTIVATE SCANNER", self.start_camera, 'Cyber.TButton'),
            ("🖼️ UPLOAD TARGET", self.upload_image, 'UltraModern.TButton'),
            ("📁 PROCESS DIRECTORY", self.process_folder, 'UltraModern.TButton'),
            ("🔴 REGISTER IDENTITY", self.register_face, 'Neon.TButton'),
            ("⏸️ PAUSE SCANNER", self.pause_camera, 'Status.TButton'),
            ("▶️ RESUME SCANNER", self.resume_camera, 'Status.TButton'),
        ]
        
        self.control_buttons = {}
        for text, command, style in buttons_config:
            btn = ttk.Button(button_container, text=text, command=command, style=style, width=25)
            btn.pack(pady=8, fill="x")
            self.control_buttons[text] = btn
            # Add hover animations
            self.add_button_hover_effect(btn)
        
        # AI Status panel
        ai_status_frame = tk.Frame(button_container, bg='#1a1a1a', relief='solid', bd=1)
        ai_status_frame.pack(fill="x", pady=(30, 0))
        
        tk.Label(ai_status_frame, text="🤖 AI STATUS", font=("Consolas", 12, "bold"), 
                fg="#00ff00", bg="#1a1a1a").pack(pady=5)
        
        self.ai_status_labels = {}
        status_items = [
            ("NEURAL NET:", "ACTIVE", "#00ff00"),
            ("FACE DETECTOR:", "READY", "#00aaff"),
            ("CLASSIFIER:", "LOADED", "#ffaa00"),
            ("CAMERA:", "STANDBY", "#ff6600")
        ]
        
        for label, status, color in status_items:
            status_row = tk.Frame(ai_status_frame, bg='#1a1a1a')
            status_row.pack(fill="x", padx=10, pady=2)
            tk.Label(status_row, text=label, font=("Consolas", 9), fg="#888888", bg="#1a1a1a").pack(side="left")
            status_lbl = tk.Label(status_row, text=status, font=("Consolas", 9, "bold"), fg=color, bg="#1a1a1a")
            status_lbl.pack(side="right")
            self.ai_status_labels[label] = status_lbl

    def add_button_hover_effect(self, button):
        """Add hover animation effects to buttons"""
        def on_enter(e):
            self.animate_button_hover(button, True)
        
        def on_leave(e):
            self.animate_button_hover(button, False)
        
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

    def animate_button_hover(self, button, entering):
        """Animate button hover effects"""
        # This would typically involve changing colors, but TTK styling limits this
        # In a real implementation, you'd use custom drawing or tkinter.Canvas
        pass

    def create_neural_display(self, parent):
        """Create the main display area with neural network styling"""
        display_frame = tk.Frame(parent, bg='#0d0d0d', relief='solid', bd=2)
        display_frame.pack(side="left", fill="both", expand=True, padx=(0, 15), pady=10)
        
        # Display header
        display_header = tk.Frame(display_frame, bg='#1a1a1a', height=50)
        display_header.pack(fill="x", padx=2, pady=2)
        display_header.pack_propagate(False)
        
        tk.Label(display_header, text="🔍 NEURAL VISION DISPLAY 🔍", 
                font=("Consolas", 14, "bold"), fg="#ff00aa", bg="#1a1a1a").pack(expand=True)
        
        # Main display area
        display_container = tk.Frame(display_frame, bg='#0d0d0d')
        display_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Camera display with border effect
        camera_border = tk.Frame(display_container, bg='#00ffff', relief='solid', bd=3)
        camera_border.pack(expand=True, fill="both")
        
        self.label_camera = tk.Label(camera_border, 
                                    text="◢ INITIALIZING NEURAL SCANNER ◣\n\n🔄 Loading AI Models...\n\n⚡ Please Wait ⚡",
                                    font=("Consolas", 16, "bold"),
                                    fg="#00ffff",
                                    bg="#000000",
                                    justify="center")
        self.label_camera.pack(expand=True, fill="both", padx=3, pady=3)
        
        # Scanning overlay frame
        self.scanning_overlay = tk.Frame(display_container, bg='#000000')
        # This will be used for scanning animations

    def create_results_terminal(self, parent):
        """Create cyberpunk-style results terminal"""
        results_frame = tk.Frame(parent, bg='#111111', relief='solid', bd=2, width=350)
        results_frame.pack(side="right", fill="y", pady=10)
        results_frame.pack_propagate(False)
        
        # Results header
        results_header = tk.Frame(results_frame, bg='#1a1a1a', height=50)
        results_header.pack(fill="x", padx=2, pady=2)
        results_header.pack_propagate(False)
        
        tk.Label(results_header, text="📊 SCAN RESULTS TERMINAL 📊", 
                font=("Consolas", 12, "bold"), fg="#00ff00", bg="#1a1a1a").pack(expand=True)
        
        # Scrollable results area
        results_container = tk.Frame(results_frame, bg='#111111')
        results_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create scrollable text widget for results
        results_scroll_frame = tk.Frame(results_container, bg='#000000', relief='sunken', bd=2)
        results_scroll_frame.pack(fill="both", expand=True)
        
        self.results_text = tk.Text(results_scroll_frame,
                                   bg='#000000',
                                   fg='#00ff00',
                                   font=("Consolas", 10),
                                   wrap="word",
                                   state="disabled",
                                   insertbackground='#00ff00',
                                   selectbackground='#333333')
        
        results_scrollbar = tk.Scrollbar(results_scroll_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")
        
        # Initialize with welcome message
        self.add_terminal_message("🚀 NEURAL FACE RECOGNITION SYSTEM INITIALIZED", "#00ffff")
        self.add_terminal_message("⚡ AI Models loaded successfully", "#00ff00")
        self.add_terminal_message("🔍 Ready for facial recognition operations", "#ffaa00")
        self.add_terminal_message("=" * 40, "#333333")

    def create_cyber_status_bar(self, parent):
        """Create animated cyberpunk status bar"""
        status_frame = tk.Frame(parent, bg='#1a1a1a', height=60, relief='raised', bd=2)
        status_frame.pack(fill="x", side="bottom")
        status_frame.pack_propagate(False)
        
        # Left status info
        left_status = tk.Frame(status_frame, bg='#1a1a1a')
        left_status.pack(side="left", fill="y", padx=20)
        
        self.status_label = tk.Label(left_status, 
                                    text="💫 SYSTEM READY", 
                                    font=("Consolas", 12, "bold"), 
                                    fg="#00ffff", 
                                    bg="#1a1a1a")
        self.status_label.pack(anchor="w", pady=5)
        
        # System info
        info_text = f"CPU: ⚡ GPU: {'�' if torch.cuda.is_available() else '❄️'} RAM: 💾 NEURAL NET: 🧠"
        self.info_label = tk.Label(left_status, 
                                  text=info_text, 
                                  font=("Consolas", 9), 
                                  fg="#888888", 
                                  bg="#1a1a1a")
        self.info_label.pack(anchor="w")
        
        # Right status indicators
        right_status = tk.Frame(status_frame, bg='#1a1a1a')
        right_status.pack(side="right", fill="y", padx=20)
        
        # Time and performance indicators
        perf_frame = tk.Frame(right_status, bg='#1a1a1a')
        perf_frame.pack(anchor="e", pady=5)
        
        self.fps_label = tk.Label(perf_frame, text="FPS: --", font=("Consolas", 9), 
                                 fg="#00ff00", bg="#1a1a1a")
        self.fps_label.pack(side="left", padx=5)
        
        self.faces_label = tk.Label(perf_frame, text="FACES: 0", font=("Consolas", 9), 
                                   fg="#ffaa00", bg="#1a1a1a")
        self.faces_label.pack(side="left", padx=5)
        
        self.confidence_label = tk.Label(perf_frame, text="CONFIDENCE: N/A", font=("Consolas", 9), 
                                        fg="#ff00aa", bg="#1a1a1a")
        self.confidence_label.pack(side="left", padx=5)

    def add_terminal_message(self, message, color="#00ff00"):
        """Add a message to the results terminal with timestamp"""
        # Only add message if GUI components exist
        if hasattr(self, 'results_text') and self.results_text:
            try:
                timestamp = time.strftime("%H:%M:%S")
                formatted_message = f"[{timestamp}] {message}\n"
                
                self.results_text.configure(state="normal")
                self.results_text.insert("end", formatted_message)
                self.results_text.configure(state="disabled")
                self.results_text.see("end")
            except Exception as e:
                logging.debug(f"Could not add terminal message: {e}")
        else:
            # Log the message instead if terminal is not ready
            logging.info(f"Terminal: {message}")

    def update_ai_status(self, component, status, color="#00ff00"):
        """Update AI component status"""
        if hasattr(self, 'ai_status_labels') and component in self.ai_status_labels:
            self.ai_status_labels[component].configure(text=status, fg=color)

    def update_performance_stats(self, fps=None, faces=None, confidence=None):
        """Update performance statistics in status bar"""
        try:
            if fps is not None and hasattr(self, 'fps_label') and self.fps_label:
                self.fps_label.configure(text=f"FPS: {fps:.1f}")
            if faces is not None and hasattr(self, 'faces_label') and self.faces_label:
                self.faces_label.configure(text=f"FACES: {faces}")
            if confidence is not None and hasattr(self, 'confidence_label') and self.confidence_label:
                if confidence > 0:
                    self.confidence_label.configure(text=f"CONFIDENCE: {confidence:.1f}%")
                else:
                    self.confidence_label.configure(text="CONFIDENCE: N/A")
        except Exception as e:
            logging.debug(f"Could not update performance stats: {e}")

    def update_status(self, message, status_type="info"):
        """Update status bar with cyberpunk styling and terminal logging"""
        status_icons = {
            "info": "💫",
            "success": "✅", 
            "warning": "⚠️",
            "error": "🚨",
            "processing": "🔄",
            "scanning": "🔍",
            "neural": "🧠"
        }
        
        status_colors = {
            "info": "#00ffff",
            "success": "#00ff00",
            "warning": "#ffaa00", 
            "error": "#ff3333",
            "processing": "#ff00aa",
            "scanning": "#00aaff",
            "neural": "#aa00ff"
        }
        
        icon = status_icons.get(status_type, "💫")
        color = status_colors.get(status_type, "#00ffff")
        
        # Only update status label if it exists
        if hasattr(self, 'status_label') and self.status_label:
            try:
                self.status_label.config(text=f"{icon} {message}", fg=color)
                self.root.update_idletasks()
            except Exception as e:
                logging.debug(f"Could not update status label: {e}")
        
        # Add to terminal (with defensive check)
        self.add_terminal_message(f"{icon} {message}", color)
        
        # Also log to console
        logging.info(f"Status: {message}")

    def register_face(self):
        """Wrapper method for opening registration window"""
        self.open_registration_window()

    def pause_camera(self):
        """Pause the camera scanning"""
        if self.running:
            self.running = False
            self.update_status("⏸️ SCANNER PAUSED", "warning")
            self.update_ai_status("CAMERA:", "PAUSED", "#ffaa00")
            self.label_camera.configure(text="⏸️ SCANNER PAUSED ⏸️\n\n🔄 Click Resume to Continue\n\n⚡ Neural Net Ready")
            if hasattr(self, 'control_buttons'):
                self.control_buttons["🎯 ACTIVATE SCANNER"].configure(text="▶️ RESUME SCANNER")

    def clear_preview(self):
        """Clear the camera preview display"""
        if hasattr(self, 'label_camera'):
            try:
                # Clear the image first
                self.label_camera.configure(image="")
                self.label_camera.imgtk = None
                # Then set the text
                self.label_camera.configure(text="◢ NEURAL SCANNER STANDBY ◣\n\n🤖 AI Ready for Analysis\n\n⚡ Awaiting Input")
            except Exception as e:
                logging.warning(f"Error clearing preview: {e}")
                # Fallback: just try to set text
                try:
                    self.label_camera.configure(image="", text="◢ NEURAL SCANNER STANDBY ◣\n\n🤖 AI Ready for Analysis\n\n⚡ Awaiting Input")
                except:
                    pass

    def clear_display_area(self):
        """Clear the results display area"""
        try:
            if hasattr(self, 'results_text'):
                self.results_text.configure(state="normal")
                self.results_text.delete("1.0", "end")
                self.results_text.configure(state="disabled")
                # Re-add header
                self.add_terminal_message("🚀 DISPLAY CLEARED - READY FOR NEW ANALYSIS", "#00ffff")
        except:
            pass

    def start_camera(self):
        """Start the neural scanner with cyberpunk effects"""
        if not self._switch_camera_mode('main'):
            return
            
        self.clear_preview()
        self._hide_resume_button()
        self.update_status("🔄 INITIALIZING NEURAL SCANNER...", "processing")
        self.update_ai_status("CAMERA:", "STARTING", "#ffaa00")

        # Check if camera is already initialized
        if self.cap is None or not self.cap.isOpened():
            self._initialize_camera()
            
        if self.cap is None or not self.cap.isOpened():
            logging.error("Failed to open webcam")
            self.update_status("🚨 CAMERA CONNECTION FAILED", "error")
            self.update_ai_status("CAMERA:", "ERROR", "#ff3333")
            messagebox.showerror("🚨 Camera Error", "Failed to open camera. Please check if it's connected and not used by another application.")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self.process_camera, daemon=True)
        self.thread.start()
        self.update_gui()
        
        # Update UI elements
        self.label_camera.configure(text="⚡ NEURAL SCANNER ACTIVE ⚡\n\n🔍 SCANNING FOR FACES...\n\n📡 Real-time Analysis")
        self.update_status("🎯 SCANNER ACTIVE - NEURAL RECOGNITION ONLINE", "success")
        self.update_ai_status("CAMERA:", "ACTIVE", "#00ff00")
        
        # Animate button states
        if hasattr(self, 'control_buttons'):
            self.control_buttons["🎯 ACTIVATE SCANNER"].configure(text="⏹️ STOP SCANNER")
            
        logging.info("Neural scanner started successfully")

    def stop_camera(self):
        self.running = False
        self.running_registration = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)  # Wait for thread to finish
        try:
            if hasattr(self, 'label_camera'):
                self.label_camera.configure(image="")
                self.label_camera.imgtk = None
        except:
            pass
        self.camera_mode = None
        # Don't release camera - keep it for faster restarts

    def process_camera(self):
        while self.running:
            try:
                self.frame_count += 1
                if self.frame_count % 3 != 0:
                    continue
                ret, frame = self.cap.read()
                if not ret:
                    continue
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resize to half
                results = predict_faces(small_frame, self.mtcnn, self.facenet, self.svm, self.encoder, self.device)
                scaled_results = [(
                    [coord * 2 for coord in box], name, conf
                ) for box, name, conf in results]
                
                # Only keep the latest frame in queue to prevent memory buildup
                if not self.queue.empty():
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass
                self.queue.put((frame, scaled_results))
            except Exception as e:
                logging.error(f"Error in camera processing: {e}")
                continue

    def update_gui(self):
        """Update the neural display with enhanced cyberpunk effects"""
        try:
            frame, results = self.queue.get_nowait()
            
            # Update scanning angle for animation
            self.scanning_angle = (self.scanning_angle + 5) % 100
            
            # Calculate FPS
            current_time = time.time()
            if hasattr(self, 'last_frame_time'):
                fps = 1.0 / (current_time - self.last_frame_time)
                self.update_performance_stats(fps=fps)
            self.last_frame_time = current_time
            
            # Process frame with enhanced visualization
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            
            # Use cyberpunk font styling
            try:
                font = ImageFont.truetype("arial.ttf", 18)
                font_large = ImageFont.truetype("arial.ttf", 24)
            except (OSError, IOError):
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 18)
                    font_large = ImageFont.truetype("DejaVuSans.ttf", 24)
                except (OSError, IOError):
                    font = ImageFont.load_default()
                    font_large = ImageFont.load_default()

            # Enhanced face detection visualization
            face_count = len(results)
            self.update_performance_stats(faces=face_count)
            
            max_confidence = 0
            for i, (box, name, conf) in enumerate(results):
                # Cyberpunk color scheme
                if name == "Unknown":
                    box_color = "#ff3333"  # Red for unknown
                    glow_color = "#ff6666"
                else:
                    box_color = "#00ffff"  # Cyan for known
                    glow_color = "#66ffff"
                    max_confidence = max(max_confidence, conf)
                
                # Draw enhanced bounding box with glow effect
                box_coords = [int(coord) for coord in box]
                
                # Outer glow
                for offset in range(3, 0, -1):
                    alpha = 100 - (offset * 30)
                    glow_box = [
                        box_coords[0] - offset, box_coords[1] - offset,
                        box_coords[2] + offset, box_coords[3] + offset
                    ]
                    draw.rectangle(glow_box, outline=glow_color, width=1)
                
                # Main box
                draw.rectangle(box_coords, outline=box_color, width=3)
                
                # Enhanced text with background
                if name == "Unknown":
                    display_text = "⚠️ UNKNOWN SUBJECT"
                    confidence_text = f"THREAT LEVEL: {conf:.1f}%"
                else:
                    display_text = f"✅ {name.upper()}"
                    confidence_text = f"MATCH: {conf:.1f}%"
                
                # Text background
                text_bbox = draw.textbbox((box_coords[0], box_coords[1] - 50), display_text, font=font)
                draw.rectangle([text_bbox[0]-5, text_bbox[1]-2, text_bbox[2]+5, text_bbox[3]+2], 
                             fill="#000000", outline=box_color)
                
                # Main text
                draw.text((box_coords[0], box_coords[1] - 45), display_text, 
                         fill=box_color, font=font)
                draw.text((box_coords[0], box_coords[1] - 25), confidence_text, 
                         fill=box_color, font=font)
                
                # Add scanning effect
                scan_y = box_coords[1] + (self.scanning_angle % (box_coords[3] - box_coords[1]))
                draw.line([box_coords[0], scan_y, box_coords[2], scan_y], 
                         fill=glow_color, width=2)

            # Add scanning overlay text
            if face_count > 0:
                overlay_text = f"🔍 SCANNING: {face_count} SUBJECT{'S' if face_count != 1 else ''} DETECTED"
            else:
                overlay_text = "🔍 SCANNING FOR SUBJECTS..."
            
            # Draw scanning overlay
            draw.text((10, 10), overlay_text, fill="#00ffff", font=font_large)
            
            # Update performance stats
            self.update_performance_stats(confidence=max_confidence if max_confidence > 0 else None)

            # Display the enhanced frame
            imgtk = ImageTk.PhotoImage(image=img.resize((800, 600)))
            self.label_camera.imgtk = imgtk
            try:
                self.label_camera.configure(image=imgtk)
            except Exception as e:
                logging.warning(f"Error setting camera image: {e}")
            
            # Add terminal updates for detections
            if results:
                for box, name, conf in results:
                    if name != "Unknown":
                        self.add_terminal_message(f"🎯 IDENTITY CONFIRMED: {name} ({conf:.1f}%)", "#00ff00")
                    else:
                        self.add_terminal_message(f"⚠️ UNKNOWN SUBJECT DETECTED ({conf:.1f}%)", "#ff3333")
                        
        except queue.Empty:
            pass
        except Exception as e:
            logging.error(f"Error in GUI update: {e}")

        if self.running:
            self.root.after(30, self.update_gui)  # Slightly slower for better performance

    def upload_image(self):
        """Upload and analyze image with cyberpunk effects"""
        # Pause camera instead of stopping it completely
        was_running = self.running
        self.running = False
        
        self.clear_preview()
        self.clear_display_area()
        self.update_status("📁 SELECTING TARGET IMAGE...", "processing")

        file_path = filedialog.askopenfilename(
            title="🎯 Select Target Image for Analysis",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            # Resume camera if it was running
            if was_running:
                self.running = True
            self.update_status("❌ TARGET SELECTION CANCELLED", "warning")
            return

        self.update_status("🔍 ANALYZING TARGET IMAGE...", "processing")
        self.add_terminal_message(f"📁 Loading image: {os.path.basename(file_path)}", "#00aaff")

        frame = cv2.imread(file_path)
        if frame is None:
            logging.error("Invalid image")
            self.update_status("🚨 INVALID IMAGE FORMAT", "error")
            self.add_terminal_message("🚨 ERROR: Invalid or corrupted image file", "#ff3333")
            # Resume camera if it was running
            if was_running:
                self.running = True
            return

        # Process image
        resized_frame = cv2.resize(frame, (800, 600))
        results = predict_faces(resized_frame, self.mtcnn, self.facenet, self.svm, self.encoder, self.device)

        # Enhanced visualization
        img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            font_large = ImageFont.truetype("arial.ttf", 28)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 20)
                font_large = ImageFont.truetype("DejaVuSans.ttf", 28)
            except (OSError, IOError):
                font = ImageFont.load_default()
                font_large = ImageFont.load_default()

        analysis_results = []
        for i, (box, name, conf) in enumerate(results):
            # Cyberpunk styling
            if name == "Unknown":
                box_color = "#ff3333"
                glow_color = "#ff6666"
                status_icon = "⚠️"
                classification = "UNKNOWN SUBJECT"
            else:
                box_color = "#00ffff"
                glow_color = "#66ffff"
                status_icon = "✅"
                classification = f"CONFIRMED: {name.upper()}"
            
            # Enhanced bounding box with glow
            box_coords = [int(coord) for coord in box]
            
            # Glow effect
            for offset in range(4, 0, -1):
                glow_box = [
                    box_coords[0] - offset, box_coords[1] - offset,
                    box_coords[2] + offset, box_coords[3] + offset
                ]
                draw.rectangle(glow_box, outline=glow_color, width=1)
            
            # Main box
            draw.rectangle(box_coords, outline=box_color, width=4)
            
            # Enhanced labels
            main_text = f"{status_icon} {classification}"
            conf_text = f"CONFIDENCE: {conf:.1f}%"
            
            # Text with background
            text_y = box_coords[1] - 60
            text_bbox = draw.textbbox((box_coords[0], text_y), main_text, font=font)
            draw.rectangle([text_bbox[0]-5, text_bbox[1]-2, text_bbox[2]+5, text_bbox[3]+25], 
                         fill="#000000", outline=box_color, width=2)
            
            draw.text((box_coords[0], text_y), main_text, fill=box_color, font=font)
            draw.text((box_coords[0], text_y + 25), conf_text, fill=box_color, font=font)
            
            analysis_results.append(f"{status_icon} {classification} - {conf:.1f}%")

        # Add analysis overlay
        overlay_text = f"🔍 ANALYSIS COMPLETE: {len(results)} FACE{'S' if len(results) != 1 else ''} DETECTED"
        draw.text((10, 10), overlay_text, fill="#00ffff", font=font_large)

        # Display enhanced image
        imgtk = ImageTk.PhotoImage(image=img)
        self.label_camera.imgtk = imgtk
        try:
            self.label_camera.configure(image=imgtk)
        except Exception as e:
            logging.warning(f"Error setting camera image: {e}")
        
        # Update terminal with results
        self.add_terminal_message("🎯 IMAGE ANALYSIS COMPLETE", "#00ff00")
        for result in analysis_results:
            color = "#00ff00" if "CONFIRMED" in result else "#ff3333"
            self.add_terminal_message(result, color)
        
        # Update status
        if results:
            known_count = sum(1 for _, name, _ in results if name != "Unknown")
            unknown_count = len(results) - known_count
            status_msg = f"✅ ANALYSIS COMPLETE: {known_count} KNOWN, {unknown_count} UNKNOWN"
            self.update_status(status_msg, "success")
        else:
            self.update_status("🔍 NO FACES DETECTED IN IMAGE", "warning")
            
        # Show resume button if camera was running
        if was_running:
            self._show_resume_button()
            self.update_status("📷 Camera paused - Click Resume to continue scanning", "info")

        self.update_status("Processing image...", "processing")
        frame = cv2.imread(file_path)
        if frame is None:
            logging.error("Invalid image")
            self.update_status("Failed to load image", "error")
            # Resume camera if it was running
            if was_running:
                self.running = True
            return

        resized_frame = cv2.resize(frame, (640, 480))
        results = predict_faces(resized_frame, self.mtcnn, self.facenet, self.svm, self.encoder, self.device)

        img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 16)
            except (OSError, IOError):
                font = ImageFont.load_default()

        prediction_texts = []
        for box, name, conf in results:
            color = "red" if name == "Unknown" else "lime"
            draw.rectangle(box, outline=color, width=2)
            text = "Unknown" if name == "Unknown" else f"{name} ({conf:.1f}%)"
            draw.text((box[0], box[1] - 20), text, fill=color, font=font)
            prediction_texts.append(text)

        # Remove placeholder text when displaying image
        self.label_camera.configure(text="")
        imgtk = ImageTk.PhotoImage(image=img)
        self.label_camera.imgtk = imgtk
        try:
            self.label_camera.configure(image=imgtk)
        except Exception as e:
            logging.warning(f"Error setting camera label image: {e}")
        
        if prediction_texts:
            if hasattr(self, 'label_prediction'):
                self.label_prediction.configure(text="🎯 Detection Results:\n" + "\n".join(prediction_texts))
            if hasattr(self, 'label_prediction'):
                self.label_prediction.configure(text="🎯 Detection Results:\n" + "\n".join(prediction_texts))
            self.update_status(f"Analysis complete - {len(results)} face(s) detected", "success")
        else:
            if hasattr(self, 'label_prediction'):
                self.label_prediction.configure(text="❌ No faces detected in the image")
            self.update_status("No faces detected", "warning")
        
        self.root.geometry("1100x900")
        
        # Show resume button if camera was running
        if was_running:
            self._show_resume_button()

        # Resume camera if it was running
        if was_running:
            self.running = True

    def process_folder(self):
        # Pause camera instead of stopping it completely
        was_running = self.running
        self.running = False
        
        self.clear_preview()
        self.clear_display_area()

        folder_path = filedialog.askdirectory()
        if not folder_path:
            # Resume camera if it was running
            if was_running:
                self.running = True
            return

        try:
            # Get list of image files first for progress tracking
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if not image_files:
                messagebox.showinfo("No Images", "No image files found in the selected folder.")
                if was_running:
                    self.running = True
                return

            embeddings = []
            image_paths = []
            face_infos = []
            processed_count = 0

            # Create modern progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("🔄 Processing Folder")
            progress_window.geometry("500x200")
            progress_window.configure(bg="#1A1A3A")
            progress_window.transient(self.root)
            progress_window.grab_set()
            progress_window.resizable(False, False)
            
            # Center the window
            progress_window.update_idletasks()
            x = (progress_window.winfo_screenwidth() // 2) - (500 // 2)
            y = (progress_window.winfo_screenheight() // 2) - (200 // 2)
            progress_window.geometry(f"500x200+{x}+{y}")
            
            # Header
            header_frame = tk.Frame(progress_window, bg="#2E2E5C", height=50)
            header_frame.pack(fill="x")
            header_frame.pack_propagate(False)
            
            header_label = tk.Label(header_frame, text="📊 Analyzing Images", 
                                  font=("Segoe UI", 14, "bold"), fg="white", bg="#2E2E5C")
            header_label.pack(pady=15)
            
            # Content area
            content_frame = tk.Frame(progress_window, bg="#1A1A3A")
            content_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            progress_var = tk.StringVar()
            progress_info = tk.Label(content_frame, textvariable=progress_var,
                                   fg="#4A90E2", bg="#1A1A3A", font=("Segoe UI", 11))
            progress_info.pack(pady=(0, 10))
            
            # Progress bar simulation
            progress_bar_frame = tk.Frame(content_frame, bg="#333333", height=20, relief="sunken", bd=1)
            progress_bar_frame.pack(fill="x", pady=(0, 15))
            
            self.progress_bar = tk.Frame(progress_bar_frame, bg="#4A90E2", height=18)
            self.progress_bar.pack(side="left", fill="y")
            
            # Cancel button
            cancel_btn = ttk.Button(content_frame, text="❌ Cancel", 
                                  command=lambda: setattr(self, '_cancel_processing', True),
                                  style='Accent.TButton')
            cancel_btn.pack()
            
            self._cancel_processing = False

            for filename in image_files:
                if self._cancel_processing:
                    break
                    
                try:
                    processed_count += 1
                    progress_var.set(f"Processing {processed_count}/{len(image_files)}: {filename}")
                    
                    # Update progress bar
                    progress_percent = processed_count / len(image_files)
                    bar_width = int(460 * progress_percent)  # 460 is approximate width
                    self.progress_bar.config(width=bar_width)
                    
                    progress_window.update()

                    image_path = os.path.join(folder_path, filename)
                    frame = cv2.imread(image_path)
                    if frame is None:
                        logging.warning(f"Could not load image: {filename}")
                        continue

                    # Limit image size to prevent memory issues
                    height, width = frame.shape[:2]
                    if width > 1024 or height > 1024:
                        scale = min(1024/width, 1024/height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))

                    resized_frame = cv2.resize(frame, (640, 480))
                    results = predict_faces(resized_frame, self.mtcnn, self.facenet, self.svm, self.encoder, self.device)

                    for box, name, conf in results:
                        face_infos.append((filename, name, conf))

                        if name == "Unknown":
                            try:
                                # Use the same method as get_faces_and_embeddings for consistency
                                faces = get_faces_and_embeddings(resized_frame, self.mtcnn, self.facenet, self.device)
                                if faces:
                                    _, embedding, _ = faces[0]  # Use first face if multiple detected
                                    embeddings.append(embedding)
                                    image_paths.append(filename)
                            except Exception as e:
                                logging.warning(f"Failed to extract embedding for {filename}: {e}")
                                continue

                except Exception as e:
                    logging.error(f"Error processing {filename}: {e}")
                    continue

            progress_window.destroy()
            
            if self._cancel_processing:
                messagebox.showinfo("Cancelled", "Processing was cancelled by user.")
                if was_running:
                    self.running = True
                return

        except Exception as e:
            logging.error(f"Error in process_folder: {e}")
            messagebox.showerror("Error", f"An error occurred while processing the folder: {e}")
            if was_running:
                self.running = True
            return

        # Clustering and grouping
        clusters = defaultdict(list)
        if embeddings:
            try:
                embeddings_array = np.array(embeddings)
                clustering = DBSCAN(eps=0.8, min_samples=1, metric='cosine').fit(embeddings_array)
                
                # Create a mapping for cluster labels to person names
                cluster_person_map = {}
                person_counter = 1
                
                for idx, label in enumerate(clustering.labels_):
                    if label >= 0:  # Valid cluster (not noise)
                        if label not in cluster_person_map:
                            cluster_person_map[label] = f"Unknown Person {person_counter}"
                            person_counter += 1
                        person_name = cluster_person_map[label]
                    else:  # Noise points
                        person_name = f"Unknown Person {person_counter}"
                        person_counter += 1
                    
                    clusters[person_name].append(image_paths[idx])
            except Exception as e:
                logging.error(f"Error in clustering: {e}")
                # Fallback: put all unknown faces in separate persons
                for idx, path in enumerate(image_paths):
                    clusters[f"Unknown Person {idx + 1}"].append(path)

        known_group = defaultdict(list)
        for fname, name, conf in face_infos:
            if name != "Unknown":
                known_group[name].append(fname)

        # Clear previous results
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        self.thumbnails = []
        
        # Create scroll canvas if it doesn't exist
        if self.canvas_frame is None:
            self.canvas_frame = tk.Frame(self.root, bg='#2c3e50')
            self.scroll_canvas = tk.Canvas(self.canvas_frame, bg='#2c3e50')
            scrollbar = tk.Scrollbar(self.canvas_frame, orient="vertical", command=self.scroll_canvas.yview)
            self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
            
            # Reset scroll_frame to be inside canvas
            self.scroll_frame = tk.Frame(self.scroll_canvas, bg='#2c3e50')
            self.scroll_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
            
            self.scroll_canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

        def create_thumbnail_label(img_path, parent):
            try:
                full_path = os.path.join(folder_path, img_path)
                frame = cv2.imread(full_path)
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame).resize((120, 120))
                    photo = ImageTk.PhotoImage(image)
                    self.thumbnails.append(photo)
                    
                    # Create thumbnail container
                    thumb_container = tk.Frame(parent, bg="#3A3A5A", relief="raised", bd=1)
                    thumb_container.pack(side="left", padx=5, pady=5)
                    
                    lbl = tk.Label(thumb_container, image=photo, bg="#3A3A5A")
                    lbl.pack(padx=2, pady=2)
                    
                    # Add filename label
                    name_label = tk.Label(thumb_container, text=img_path[:15] + "..." if len(img_path) > 15 else img_path,
                                        fg="white", bg="#3A3A5A", font=("Segoe UI", 8))
                    name_label.pack(pady=(0, 2))
            except Exception as e:
                logging.warning(f"Failed to create thumbnail for {img_path}: {e}")

        # Display known faces first
        for person, files in known_group.items():
            group = tk.LabelFrame(self.scroll_frame, 
                                text=f"👤 {person} ({len(files)} images)", 
                                fg="#4A90E2", 
                                bg="#2A2A4A", 
                                font=("Segoe UI", 11, "bold"),
                                relief="ridge",
                                bd=2)
            group.pack(fill="x", pady=8, padx=10)
            
            # Create thumbnail container
            thumb_frame = tk.Frame(group, bg="#2A2A4A")
            thumb_frame.pack(fill="x", padx=10, pady=10)
            
            for fname in files:
                create_thumbnail_label(fname, thumb_frame)

        # Display unknown face clusters
        for cluster_label, files in clusters.items():
            # Use a person icon for unknown persons instead of question mark
            icon = "👤" if "Unknown Person" in cluster_label else "❓"
            group = tk.LabelFrame(self.scroll_frame, 
                                text=f"{icon} {cluster_label} ({len(files)} images)", 
                                fg="#E67E22", 
                                bg="#2A2A4A", 
                                font=("Segoe UI", 11, "bold"),
                                relief="ridge",
                                bd=2)
            group.pack(fill="x", pady=8, padx=10)
            
            # Create thumbnail container
            thumb_frame = tk.Frame(group, bg="#2A2A4A")
            thumb_frame.pack(fill="x", padx=10, pady=10)
            
            for fname in files:
                create_thumbnail_label(fname, thumb_frame)

        # Display summary
        if not face_infos and not embeddings:
            msg = tk.Label(self.scroll_frame, text="No faces detected in the selected folder.",
                        fg="white", bg="#2A2A3D", font=("Helvetica", 12))
            msg.pack(pady=20)
        else:
            total_faces = len(face_infos)
            known_faces = len([f for f in face_infos if f[1] != "Unknown"])
            unknown_faces = len(embeddings);
            
            summary_text = (f"Processed {len(image_files)} images\n"
                          f"Found {total_faces} faces total\n"
                          f"Known: {known_faces}, Unknown: {unknown_faces}");
            
            summary_msg = tk.Label(self.scroll_frame, text=summary_text,
                                fg="lightgreen", bg="#2A2A3D", font=("Helvetica", 11))
            summary_msg.pack(pady=10)

        # Update scroll region and display
        if self.scroll_canvas:
            self.scroll_canvas.update_idletasks()
            self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))
            
        if self.canvas_frame:
            self.canvas_frame.pack(fill="both", expand=True)
        self.root.geometry("1000x800")

        # Resume camera if it was running
        if was_running:
            self.running = True
            self._show_resume_button()

        messagebox.showinfo("Processing Complete", 
                          f"Processed folder: {os.path.basename(folder_path)}\n"
                          f"Found {len(face_infos)} faces in {len(image_files)} images")
        self.root.geometry("1000x800")

    def open_registration_window(self):
        """Open cyberpunk-style registration window"""
        if hasattr(self, 'reg_window') and self.reg_window.winfo_exists():
            self.close_registration_window()

        # Switch to registration mode without stopping camera completely
        if not self._switch_camera_mode('registration'):
            return
            
        # Ensure camera is available
        if self.cap is None or not self.cap.isOpened():
            self._initialize_camera()
            
        if self.cap is None or not self.cap.isOpened():
            messagebox.showerror("🚨 Camera Error", "Failed to access camera for registration.")
            return

        self.update_status("🔴 OPENING IDENTITY REGISTRATION", "processing")
        
        self.clear_preview()
        self.running_registration = True
        self.capture_count = 0
        self.max_captures = 60
        self.collected_frames = []

        # Create cyberpunk registration window
        self.reg_window = tk.Toplevel(self.root)
        self.reg_window.title("🔴 NEURAL IDENTITY REGISTRATION")
        self.reg_window.geometry("600x700")
        self.reg_window.configure(bg="#0a0a0a")
        self.reg_window.transient(self.root)
        
        # Header
        header = tk.Frame(self.reg_window, bg="#1a1a1a", height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        tk.Label(header, text="🔴 REGISTER NEW IDENTITY", 
                font=("Consolas", 16, "bold"), fg="#ff00aa", bg="#1a1a1a").pack(pady=15)

        # Identity selection
        identity_frame = tk.Frame(self.reg_window, bg="#0a0a0a")
        identity_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Label(identity_frame, text="SELECT OR CREATE IDENTITY:", 
                bg="#0a0a0a", fg="#00ffff", font=("Consolas", 12, "bold")).pack(anchor="w")

        self.identity_var = tk.StringVar()
        self.identity_listbox = tk.Listbox(identity_frame, 
                                          listvariable=self.identity_var, 
                                          selectmode="single", 
                                          height=4,
                                          bg="#1a1a1a", 
                                          fg="#00ff00", 
                                          font=("Consolas", 11),
                                          selectbackground="#333333")
        self.identity_listbox.pack(fill="x", pady=5)

        # Load existing identities
        dataset_dir = "images dataset"
        os.makedirs(dataset_dir, exist_ok=True)
        identities = sorted(next(os.walk(dataset_dir))[1]) if os.path.exists(dataset_dir) else []
        
        for name in identities:
            self.identity_listbox.insert(tk.END, name)

        # Add new person button
        add_btn = tk.Button(identity_frame, 
                           text="➕ CREATE NEW IDENTITY", 
                           command=self.add_new_person,
                           bg="#ff00aa", fg="white", font=("Consolas", 10, "bold"),
                           relief="flat", padx=15, pady=5)
        add_btn.pack(pady=10)

        # Registration controls
        controls_frame = tk.Frame(self.reg_window, bg="#0a0a0a")
        controls_frame.pack(fill="x", padx=20, pady=10)

        self.btn_capture = tk.Button(controls_frame, 
                                    text="🎯 START NEURAL CAPTURE", 
                                    command=self.capture_and_register,
                                    bg="#00ffff", fg="black", font=("Consolas", 12, "bold"),
                                    relief="flat", padx=20, pady=10)
        self.btn_capture.pack()

        # Progress display
        self.progress_label = tk.Label(controls_frame, 
                                      text="⚡ READY FOR CAPTURE: 0 / 60", 
                                      fg="#ffaa00", bg="#0a0a0a", font=("Consolas", 11, "bold"))
        self.progress_label.pack(pady=10)

        # Video preview
        video_frame = tk.Frame(self.reg_window, bg="#1a1a1a", relief="solid", bd=2)
        video_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.reg_video_label = tk.Label(video_frame, 
                                       text="🔍 NEURAL SCANNER PREVIEW\n\n📡 Initializing...", 
                                       bg="#000000", fg="#00ffff", 
                                       font=("Consolas", 14, "bold"),
                                       justify="center")
        self.reg_video_label.pack(expand=True, fill="both", padx=5, pady=5)

        # Start preview
        self.update_registration_preview()

        self.reg_window.protocol("WM_DELETE_WINDOW", self.close_registration_window)

    def close_registration_window(self):
        """Close registration window"""
        self.running_registration = False
        try:
            if hasattr(self, 'reg_video_label'):
                self.reg_video_label.configure(image="")
                self.reg_video_label.imgtk = None
        except Exception as e:
            logging.debug(f"Error clearing video label: {e}")
        try:
            if hasattr(self, 'reg_window'):
                self.reg_window.destroy()
        except Exception as e:
            logging.debug(f"Error destroying window: {e}")
        self.camera_mode = None
        self.update_status("🔴 REGISTRATION CANCELLED", "warning")

    def add_new_person(self):
        """Add new person for registration"""
        new_name = simpledialog.askstring("🆕 New Identity", "Enter the new person's name:")
        if new_name:
            # Sanitize name
            new_name = "".join(c for c in new_name if c.isalnum() or c in (' ', '-', '_')).strip()
            
            dataset_dir = "images dataset"
            new_dir = os.path.join(dataset_dir, new_name)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                self.identity_listbox.insert(tk.END, new_name)
                # Select the new name
                self.identity_listbox.selection_clear(0, tk.END)
                self.identity_listbox.selection_set(tk.END)
                messagebox.showinfo("✅ Success", f"Created identity '{new_name}'. Ready for registration.")
            else:
                messagebox.showinfo("ℹ️ Info", f"'{new_name}' already exists.")

    def update_registration_preview(self):
        """Update registration window camera preview"""
        if self.running_registration and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Add cyberpunk overlay
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img)
                
                # Add scanning grid
                width, height = img.size
                for i in range(0, width, 50):
                    draw.line([(i, 0), (i, height)], fill="#333333", width=1)
                for i in range(0, height, 50):
                    draw.line([(0, i), (width, i)], fill="#333333", width=1)
                
                # Add frame overlay
                draw.text((10, 10), "🔍 REGISTRATION SCAN", fill="#00ffff", font=None)
                
                imgtk = ImageTk.PhotoImage(image=img.resize((480, 360)))
                if hasattr(self, 'reg_video_label'):
                    self.reg_video_label.imgtk = imgtk
                    try:
                        self.reg_video_label.configure(image=imgtk)
                    except Exception as e:
                        logging.warning(f"Error setting registration video image: {e}")
            
        if hasattr(self, 'reg_window') and self.reg_window.winfo_exists() and self.running_registration:
            self.reg_window.after(30, self.update_registration_preview)

    def capture_and_register(self):
        """Capture and register face with enhanced security and duplication check"""
        if not hasattr(self, 'reg_window') or not self.reg_window.winfo_exists():
            return
            
        selected = self.identity_listbox.curselection()
        if not selected:
            messagebox.showwarning("⚠️ Warning", "Please select or create an identity first.")
            return

        name = self.identity_listbox.get(selected[0])
        if name == "New Person":
            name = simpledialog.askstring("🆔 New Identity", "Enter new person name:")
            if not name:
                return
            new_dir = os.path.join("images dataset", name)
            os.makedirs(new_dir, exist_ok=True)
            self.identity_listbox.insert(tk.END, name)

        if not name:
            messagebox.showerror("❌ Error", "Please select or enter a name.")
            return

        # Update progress safely
        if hasattr(self, 'progress_label') and self.progress_label.winfo_exists():
            self.progress_label.config(text="🔍 CHECKING IDENTITY...")
            self.reg_window.update_idletasks()

        # STEP 1: Capture 1 frame for identity duplication check
        try:
            # Update button safely
            if hasattr(self, 'btn_capture') and self.btn_capture.winfo_exists():
                self.btn_capture.config(state="disabled", text="🔄 ANALYZING...")

            while True:
                if not hasattr(self, 'reg_window') or not self.reg_window.winfo_exists():
                    return
                    
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Update preview safely
                if hasattr(self, 'reg_video_label') and self.reg_video_label.winfo_exists():
                    try:
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        imgtk = ImageTk.PhotoImage(image=img.resize((480, 360)))
                        self.reg_video_label.imgtk = imgtk
                        self.reg_video_label.configure(image=imgtk)
                    except Exception as e:
                        logging.warning(f"Error updating registration preview: {e}")

                if hasattr(self, 'reg_window') and self.reg_window.winfo_exists():
                    self.reg_window.update()

                resized_frame = cv2.resize(frame, (640, 480))
                faces = get_faces_and_embeddings(resized_frame, self.mtcnn, self.facenet, self.device)
                if faces:
                    _, embedding, _ = faces[0]
                    norm_embedding = Normalizer(norm='l2').transform([embedding])

                    # Check for duplication with existing faces
                    if os.path.exists("embeddings.npy") and os.path.exists("labels.npy"):
                        all_embeds = np.load("embeddings.npy")
                        all_labels = np.load("labels.npy")

                        sims = cosine_similarity(norm_embedding, all_embeds)
                        max_sim = np.max(sims)
                        matched_index = np.argmax(sims)
                        matched_label = all_labels[matched_index]

                        if max_sim > 0.65 and matched_label != name:
                            messagebox.showwarning(
                                "⚠️ Duplicate Detected",
                                f"This face already matches '{matched_label}' ({max_sim*100:.1f}%).\n"
                                f"Registration cancelled to avoid identity conflict."
                            )
                            return
                    break

            # STEP 2: Proceed to capture 60 samples
            if hasattr(self, 'progress_label') and self.progress_label.winfo_exists():
                self.progress_label.config(text="⚡ CAPTURING FACE SAMPLES...")
                self.reg_window.update_idletasks()

            collected = 0
            embeddings = []
            captured_frames = []

            while collected < 60 and hasattr(self, 'reg_window') and self.reg_window.winfo_exists():
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Update preview safely
                if hasattr(self, 'reg_video_label') and self.reg_video_label.winfo_exists():
                    try:
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        imgtk = ImageTk.PhotoImage(image=img.resize((480, 360)))
                        self.reg_video_label.imgtk = imgtk
                        self.reg_video_label.configure(image=imgtk)
                    except Exception as e:
                        logging.warning(f"Error updating registration preview: {e}")

                resized_frame = cv2.resize(frame, (640, 480))
                faces = get_faces_and_embeddings(resized_frame, self.mtcnn, self.facenet, self.device)
                if faces:
                    _, embedding, _ = faces[0]
                    norm_emb = Normalizer(norm='l2').transform([embedding])
                    embeddings.append(norm_emb[0])
                    captured_frames.append(frame.copy())
                    collected += 1
                    
                    # Update progress safely
                    if hasattr(self, 'progress_label') and self.progress_label.winfo_exists():
                        self.progress_label.config(text=f"⚡ PROGRESS: {collected} / 60")
                        self.reg_window.update_idletasks()

                if hasattr(self, 'reg_window') and self.reg_window.winfo_exists():
                    self.reg_window.update()

            embeddings = np.array(embeddings)

            # STEP 3: Save captured images
            images_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in captured_frames]
            save_captured_images(images_rgb, name)

            # STEP 4: Append new data
            if os.path.exists("embeddings.npy") and os.path.exists("labels.npy"):
                all_embeds = np.load("embeddings.npy")
                all_labels = np.load("labels.npy")
            else:
                all_embeds = np.empty((0, 512), dtype=np.float32)
                all_labels = np.array([], dtype=str)

            all_embeds = np.vstack([all_embeds, embeddings])
            all_labels = np.append(all_labels, [name] * len(embeddings))
            np.save("embeddings.npy", all_embeds)
            np.save("labels.npy", all_labels)

            # STEP 5: Retrain recognizer
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(all_labels)

            if len(set(encoded_labels)) < 2:
                messagebox.showinfo("ℹ️ Info", f"Face for '{name}' saved, but at least 2 identities are needed to train the recognizer.")
            else:
                svm = SVC(kernel='linear', probability=True)
                svm.fit(all_embeds, encoded_labels)
                joblib.dump(svm, "svm_model.pkl")
                joblib.dump(label_encoder, "label_encoder.pkl")
                self.svm, self.encoder = load_model()

            messagebox.showinfo("✅ Success", f"Registered and saved {len(embeddings)} samples for '{name}' successfully.")
            
            # Close registration window
            self.close_registration_window()
                
        except Exception as e:
            logging.error(f"Registration error: {e}")
            messagebox.showerror("❌ Error", f"Registration failed: {str(e)}")
        finally:
            # Update button safely
            if hasattr(self, 'btn_capture') and self.btn_capture.winfo_exists():
                try:
                    self.btn_capture.config(state="normal", text="🎯 START NEURAL CAPTURE")
                except Exception as e:
                    logging.debug(f"Could not update button state: {e}")

    def update_model_with_new_data(self, embeddings, name):
        """Update the SVM model with new training data"""
        try:
            embeddings = np.array(embeddings)

            # Load existing data
            if os.path.exists("embeddings.npy") and os.path.exists("labels.npy"):
                all_embeds = np.load("embeddings.npy")
                all_labels = np.load("labels.npy")
            else:
                all_embeds = np.empty((0, 512), dtype=np.float32)
                all_labels = np.array([], dtype=str)

            # Append new data
            all_embeds = np.vstack([all_embeds, embeddings])
            all_labels = np.append(all_labels, [name] * len(embeddings))
            
            # Save updated data
            np.save("embeddings.npy", all_embeds)
            np.save("labels.npy", all_labels)

            # Retrain model
            if len(set(all_labels)) >= 2:
                label_encoder = LabelEncoder()
                encoded_labels = label_encoder.fit_transform(all_labels)

                svm = SVC(kernel='linear', probability=True)
                svm.fit(all_embeds, encoded_labels)
                
                joblib.dump(svm, "svm_model.pkl")
                joblib.dump(label_encoder, "label_encoder.pkl")
                
                # Reload model
                self.svm, self.encoder = load_model()
                
                self.update_status("🧠 NEURAL NETWORK UPDATED", "success")
                self.add_terminal_message(f"🎯 NEW IDENTITY REGISTERED: {name}", "#00ff00")
                
        except Exception as e:
            logging.error(f"Model update error: {e}")
            raise

    def quit(self):
        """Clean shutdown of the application"""
        self.running = False
        self.running_registration = False
        
        # Stop animations
        self.animation_frame = -1
        
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Wait for threads to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        self.root.quit()
        self.root.destroy()

    def _initialize_camera(self):
        """Initialize camera connection"""
        try:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    # Set camera properties for better performance
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    logging.info("Camera initialized successfully")
                else:
                    logging.error("Failed to initialize camera")
        except Exception as e:
            logging.error(f"Error initializing camera: {e}")
            self.cap = None

    def _switch_camera_mode(self, mode):
        """Switch camera mode between 'main' and 'registration'"""
        if self.camera_mode == mode:
            return True
            
        # Pause current operations
        old_running = self.running
        old_running_reg = self.running_registration
        self.running = False
        self.running_registration = False
        
        # Switch mode
        self.camera_mode = mode
        
        if mode == 'main':
            self.running = old_running
        elif mode == 'registration':
            self.running_registration = True
            
        return True

    def _show_resume_button(self):
        """Show resume button (placeholder - implement if needed)"""
        # This method is called but not essential for basic functionality
        pass

    def _hide_resume_button(self):
        """Hide resume button (placeholder - implement if needed)"""
        # This method is called but not essential for basic functionality  
        pass

    def resume_camera(self):
        """Resume camera operations"""
        if not self.running:
            self.running = True
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self.process_camera, daemon=True)
                self.thread.start()
            self.update_gui()
            self.update_status("📹 Camera resumed", "success")

    def _check_face_duplicate(self, new_embedding, threshold=0.75):
        """Check if the face is already registered using cosine similarity"""
        try:
            # Load existing embeddings if available
            if os.path.exists("embeddings.npy"):
                existing_embeddings = np.load("embeddings.npy")
                
                # Normalize the new embedding
                new_embedding_norm = Normalizer(norm='l2').transform([new_embedding])[0]
                
                # Calculate similarities with all existing embeddings
                similarities = cosine_similarity([new_embedding_norm], existing_embeddings)[0]
                
                # Check if any similarity exceeds threshold
                max_similarity = np.max(similarities) if len(similarities) > 0 else 0
                
                if max_similarity > threshold:
                    logging.info(f"Duplicate detected with similarity: {max_similarity:.3f}")
                    return True
                    
            return False
            
        except Exception as e:
            logging.error(f"Error in duplicate check: {e}")
            return False  # Continue with registration if check fails

if __name__ == '__main__':
    try:
        root = tk.Tk()
        app = FaceRecognitionApp(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        messagebox.showerror("Error", f"Failed to start application: {e}")
