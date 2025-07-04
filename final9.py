import logging
import torch
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_models(image_size=160, margin=20, pretrained='vggface2'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=image_size, margin=margin, keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained=pretrained).eval().to(device)
    return mtcnn, facenet, device

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
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, probs = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return []

    faces = []
    for i, (box, prob) in enumerate(zip(boxes, probs)):
        if prob >= min_conf:
            face = img.crop(box)
            face = mtcnn.extract(img, [box], save_path=None)[0].unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = facenet(face).cpu().numpy()
            faces.append((box, embedding.flatten(), prob))
    return faces

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
        except:
            identity, confidence = 'Unknown', 0.0
        results.append((box, identity, confidence))
    return results

def save_captured_images(images, name, dataset_dir="images dataset"):
    import os
    from PIL import Image

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

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")
        self.root.configure(bg='#1E1E2F')
        self.root.geometry("800x600")
        self.mtcnn, self.facenet, self.device = setup_models()
        self.svm, self.encoder = load_model()
        if self.svm is None:
            root.quit()
            return
        self.frame_count = 0
        self.running = False
        self.cap = None
        self.queue = queue.Queue()
        self.thread = None

        self.label_title = tk.Label(root, text="Choose an Option", font=("Helvetica", 18, "bold"), fg="white", bg="#1E1E2F")
        self.label_title.pack(pady=20)

        button_frame = ttk.Frame(root)
        button_frame.pack(pady=10)

        self.btn_cam = ttk.Button(button_frame, text="Start Webcam", command=self.start_camera)
        self.btn_cam.pack(side="left", padx=5)

        self.btn_upload = ttk.Button(button_frame, text="Upload Image", command=self.upload_image)
        self.btn_upload.pack(side="left", padx=5)

        self.btn_folder = ttk.Button(button_frame, text="Group Faces from Folder", command=self.process_folder)
        self.btn_folder.pack(side="left", padx=5)

        self.btn_register = ttk.Button(button_frame, text="Register New Face", command=self.open_registration_window)
        self.btn_register.pack(side="left", padx=5)


        self.label_video = tk.Label(root, bg='#1E1E2F')
        self.label_video.pack(padx=10, pady=10)

        self.label_prediction = tk.Label(root, text="", font=("Helvetica", 14), fg="white", bg="#1E1E2F")
        self.label_prediction.pack(pady=10)

        self.canvas_frame = tk.Frame(root, bg="#1E1E2F")
        self.canvas_frame.pack_forget()
        self.scroll_canvas = tk.Canvas(self.canvas_frame, bg="#1E1E2F", highlightthickness=0)
        self.scroll_frame = tk.Frame(self.scroll_canvas, bg="#1E1E2F")
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas_window = self.scroll_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.scroll_frame.bind("<Configure>", lambda event: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))

        self.btn_exit = ttk.Button(root, text="Exit", command=self.quit)
        self.btn_exit.pack(pady=10)
    
    def clear_preview(self):
        self.label_video.configure(image=None)
        self.label_video.imgtk = None
        self.label_prediction.configure(text="")

    def clear_display_area(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        self.canvas_frame.pack_forget()

    def start_camera(self):
        self.stop_camera()
        self.clear_preview()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.error("Failed to open webcam")
            return
        self.running = True
        self.thread = threading.Thread(target=self.process_camera, daemon=True)
        self.thread.start()
        self.update_gui()
        self.label_prediction.configure(text="")

    def stop_camera(self):
        self.running = False
        self.label_video.configure(image=None)
        self.label_video.imgtk = None
        if self.cap:
            self.cap.release()
            self.cap = None

    def process_camera(self):
        while self.running:
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
            self.queue.put((frame, scaled_results))

    def update_gui(self):
        try:
            frame, results = self.queue.get_nowait()
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("arial.ttf", 16) if hasattr(ImageFont, 'truetype') else ImageFont.load_default()

            for box, name, conf in results:
                color = "red" if name == "Unknown" else "lime"
                draw.rectangle(box, outline=color, width=2)
                text = f"{name} ({conf:.1f}%)"
                draw.text((box[0], box[1] - 20), text, fill=color, font=font)


            imgtk = ImageTk.PhotoImage(image=img.resize((640, 480)))
            self.label_video.imgtk = imgtk
            self.label_video.configure(image=imgtk)
        except queue.Empty:
            pass

        if self.running:
            self.root.after(20, self.update_gui)

    def upload_image(self):
        self.stop_camera()
        self.clear_preview()
        self.clear_display_area()

        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        frame = cv2.imread(file_path)
        if frame is None:
            logging.error("Invalid image")
            return

        resized_frame = cv2.resize(frame, (640, 480))
        results = predict_faces(resized_frame, self.mtcnn, self.facenet, self.svm, self.encoder, self.device)

        img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 16) if hasattr(ImageFont, 'truetype') else ImageFont.load_default()

        prediction_texts = []
        for box, name, conf in results:
            color = "red" if name == "Unknown" else "lime"
            draw.rectangle(box.tolist(), outline=color, width=2)
            text = "Unknown" if name == "Unknown" else f"{name} ({conf:.1f}%)"
            draw.text((box[0], box[1] - 20), text, fill=color, font=font)
            prediction_texts.append(text)

        imgtk = ImageTk.PhotoImage(image=img)
        self.label_video.imgtk = imgtk
        self.label_video.configure(image=imgtk)
        self.label_prediction.configure(text="\n".join(prediction_texts))
        self.root.geometry("1000x800")

    def process_folder(self):
        self.stop_camera()
        self.clear_preview()
        self.clear_display_area()

        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        embeddings = []
        image_paths = []
        face_infos = []

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(folder_path, filename)
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            resized_frame = cv2.resize(frame, (640, 480))
            results = predict_faces(resized_frame, self.mtcnn, self.facenet, self.svm, self.encoder, self.device)

            for box, name, conf in results:
                face_infos.append((filename, name, conf))

                if name == "Unknown":
                    img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                    face_crop = img.crop(box)
                    face_crop = face_crop.resize((160, 160))
                    face_np = np.array(face_crop).astype(np.float32) / 255.0
                    face_np = (face_np - 0.5) / 0.5
                    face_np = np.transpose(face_np, (2, 0, 1))
                    face_tensor = torch.tensor(face_np).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        embedding = self.facenet(face_tensor).cpu().numpy()
                    embeddings.append(embedding.flatten())
                    image_paths.append(filename)

        clusters = defaultdict(list)
        if embeddings:
            clustering = DBSCAN(eps=0.8, min_samples=1, metric='cosine').fit(embeddings)
            for idx, label in enumerate(clustering.labels_):
                clusters[f"Unknown Cluster {label+1}"].append(image_paths[idx])

        known_group = defaultdict(list)
        for fname, name, conf in face_infos:
            if name != "Unknown":
                known_group[name].append(fname)

        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        self.thumbnails = []

        def create_thumbnail_label(img_path, parent):
            frame = cv2.imread(img_path)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame).resize((100, 100))
                photo = ImageTk.PhotoImage(image)
                self.thumbnails.append(photo)
                lbl = tk.Label(parent, image=photo, bg="#2A2A3D")
                lbl.pack(side="left", padx=5, pady=5)

        for person, files in known_group.items():
            group = tk.LabelFrame(self.scroll_frame, text=person, fg="white", bg="#2A2A3D", font=("Helvetica", 12, "bold"))
            group.pack(fill="x", pady=5, padx=10)
            for fname in files:
                create_thumbnail_label(os.path.join(folder_path, fname), group)

        for label, files in clusters.items():
            group = tk.LabelFrame(self.scroll_frame, text=label, fg="white", bg="#2A2A3D", font=("Helvetica", 12, "bold"))
            group.pack(fill="x", pady=5, padx=10)
            for fname in files:
                create_thumbnail_label(os.path.join(folder_path, fname), group)

        if not face_infos and not embeddings:
            msg = tk.Label(self.scroll_frame, text="No faces detected in the selected folder.",
                        fg="white", bg="#2A2A3D", font=("Helvetica", 12))
            msg.pack(pady=20)
        else:
            summary_msg = tk.Label(self.scroll_frame, text=f"Processed {len(face_infos) + len(embeddings)} image(s).",
                                fg="lightgreen", bg="#2A2A3D", font=("Helvetica", 12, "italic"))
            summary_msg.pack(pady=10)

        self.scroll_canvas.update_idletasks()
        messagebox.showinfo("Processing Complete", f"Processed folder: {os.path.basename(folder_path)}")


        self.canvas_frame.pack(fill="both", expand=True)
        self.root.geometry("1000x800")

    def update_registration_preview(self):
        if self.running_registration:
            ret, frame = self.reg_cap.read()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img.resize((480, 360)))
                self.reg_video_label.imgtk = imgtk
                self.reg_video_label.configure(image=imgtk)
            self.reg_window.after(20, self.update_registration_preview)

    def close_registration_window(self):
        self.running_registration = False
        if self.reg_cap:
            self.reg_cap.release()
            self.reg_cap = None
        self.reg_video_label.configure(image=None)
        self.reg_video_label.imgtk = None
        self.reg_window.destroy()

    def open_registration_window(self):
        if hasattr(self, 'reg_window') and self.reg_window.winfo_exists():
            self.close_registration_window()

        if hasattr(self, 'btn_register'):
            self.btn_register.config(state="disabled")
            self.root.after(1000, lambda: self.btn_register.config(state="normal")) 
        
        self.stop_camera()
        self.clear_preview()

        self.running_registration = True
        self.capture_count = 0
        self.max_captures = 20
        self.collected_frames = []

        self.reg_window = tk.Toplevel(self.root)
        self.reg_window.title("Register Face")
        self.reg_window.geometry("500x600")
        self.reg_window.configure(bg="#1E1E2F")

        tk.Label(self.reg_window, text="Select Identity:", bg="#1E1E2F", fg="white", font=("Helvetica", 12)).pack(pady=5)

        self.identity_var = tk.StringVar()
        self.identity_listbox = tk.Listbox(self.reg_window, listvariable=self.identity_var, selectmode="single", height=6, bg="#2A2A3D", fg="white", font=("Helvetica", 11))
        self.identity_listbox.pack(pady=5, fill="x", padx=10)

        add_btn = ttk.Button(self.reg_window, text="➕ Add New Person", command=self.add_new_person)
        add_btn.pack(pady=(0, 10))

        dataset_dir = "images dataset"
        os.makedirs(dataset_dir, exist_ok=True)
        identities = sorted(next(os.walk(dataset_dir))[1])
        if "New Person" not in identities:
            identities.append("New Person")


        for name in identities:
            self.identity_listbox.insert(tk.END, name)

        self.btn_capture = ttk.Button(self.reg_window, text="Start Capturing", command=self.capture_and_register)
        self.btn_capture.pack(pady=10)

        self.progress_label = tk.Label(self.reg_window, text="Captured: 0 / 20", fg="white", bg="#1E1E2F", font=("Helvetica", 11))
        self.progress_label.pack(pady=5)

        self.reg_video_label = tk.Label(self.reg_window, bg="#1E1E2F")
        self.reg_video_label.pack(pady=10)

        self.reg_cap = cv2.VideoCapture(0)
        self.update_registration_preview()

        self.reg_window.protocol("WM_DELETE_WINDOW", self.close_registration_window)

    def add_new_person(self):
        new_name = simpledialog.askstring("New Person", "Enter the new person's name:")
        if new_name:
            dataset_dir = "images dataset"
            new_dir = os.path.join(dataset_dir, new_name)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                self.identity_listbox.insert(tk.END, new_name)
                messagebox.showinfo("Success", f"Created identity '{new_name}'. Select it to start registration.")
            else:
                messagebox.showinfo("Info", f"'{new_name}' already exists.")

    def ask_new_name(self):
        new_name = simpledialog.askstring("New Name", "Enter new name:")
        if new_name:
            current = list(self.name_combo['values'])
            if new_name not in current:
                current.append(new_name)
                self.name_combo['values'] = current
            self.selected_name.set(new_name)

    def capture_and_register(self):
        selected = self.identity_listbox.curselection()
        if not selected:
            messagebox.showwarning("No Identity Selected", "Please select or create a person label.")
            return

        name = self.identity_listbox.get(selected[0])
        if name == "New Person":
            name = simpledialog.askstring("New Person", "Enter new person name:")
            if not name:
                return
            new_dir = os.path.join("images dataset", name)
            os.makedirs(new_dir, exist_ok=True)
            self.identity_listbox.insert(tk.END, name)

        if not name:
            messagebox.showerror("Error", "Please select or enter a name.")
            return

        self.progress_label.config(text="Checking identity...")
        self.reg_window.update_idletasks()

        # STEP 1: Capture 1 frame for identity check
        while True:
            ret, frame = self.reg_cap.read()
            if not ret:
                continue

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img.resize((480, 360)))
            self.reg_video_label.imgtk = imgtk
            self.reg_video_label.configure(image=imgtk)
            self.reg_window.update()

            resized_frame = cv2.resize(frame, (640, 480))
            faces = get_faces_and_embeddings(resized_frame, self.mtcnn, self.facenet, self.device)
            if faces:
                _, embedding, _ = faces[0]
                norm_embedding = Normalizer(norm='l2').transform([embedding])

                if os.path.exists("embeddings.npy") and os.path.exists("labels.npy"):
                    all_embeds = np.load("embeddings.npy")
                    all_labels = np.load("labels.npy")

                    from sklearn.metrics.pairwise import cosine_similarity
                    sims = cosine_similarity(norm_embedding, all_embeds)
                    max_sim = np.max(sims)
                    matched_index = np.argmax(sims)
                    matched_label = all_labels[matched_index]

                    if max_sim > 0.65 and matched_label != name:
                        messagebox.showwarning(
                            "Duplicate Detected",
                            f"This face already matches '{matched_label}' ({max_sim*100:.1f}%).\n"
                            f"Registration cancelled to avoid identity conflict."
                        )
                        return
                break

        # STEP 2: Proceed to capture 60 samples
        self.progress_label.config(text="Capturing face samples...")
        self.reg_window.update_idletasks()

        collected = 0
        embeddings = []
        captured_frames = []

        while collected < 60:
            ret, frame = self.reg_cap.read()
            if not ret:
                continue

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img.resize((480, 360)))
            self.reg_video_label.imgtk = imgtk
            self.reg_video_label.configure(image=imgtk)

            resized_frame = cv2.resize(frame, (640, 480))
            faces = get_faces_and_embeddings(resized_frame, self.mtcnn, self.facenet, self.device)
            if faces:
                _, embedding, _ = faces[0]
                norm_emb = Normalizer(norm='l2').transform([embedding])
                embeddings.append(norm_emb[0])
                captured_frames.append(frame.copy())
                collected += 1
                self.progress_label.config(text=f"Progress: {collected} / 60")
                self.reg_window.update_idletasks()

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
            messagebox.showinfo("Info", f"Face for '{name}' saved, but at least 2 identities are needed to train the recognizer.")
        else:
            svm = SVC(kernel='linear', probability=True)
            svm.fit(all_embeds, encoded_labels)
            joblib.dump(svm, "svm_model.pkl")
            joblib.dump(label_encoder, "label_encoder.pkl")
            self.svm, self.encoder = load_model()

        messagebox.showinfo("Success", f"Registered and saved {len(embeddings)} samples for '{name}' successfully.")

    def quit(self):
        self.stop_camera()
        self.root.quit()

if __name__ == '__main__':
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
