from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter

# Fungsi untuk memuat model YOLO (menggunakan cache untuk efisiensi)
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Fungsi untuk memproses dan menampilkan hasil deteksi
def display_results(image, results, confidence_threshold=0.5):
    boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    scores = results.boxes.conf.cpu().numpy()  # Confidence scores
    labels = results.boxes.cls.cpu().numpy()  # Class indices
    names = results.names  # Class names
    
    detected_objects = []
    
    for i in range(len(boxes)):
        if scores[i] > confidence_threshold:  # Threshold untuk confidence
            x1, y1, x2, y2 = boxes[i].astype(int)
            label = names[int(labels[i])]
            score = scores[i]
            detected_objects.append(label)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, detected_objects

# Fungsi utama Streamlit
def main():
    st.set_page_config(page_title="YOLO 11 Deteksi Objek", layout="wide")
    st.title("🔍selamat-datang deteksi objeck real-time hendri")
    st.sidebar.title("⚙ *Pengaturan*")
    
    # Load model YOLO
    model_path = "yolo11n.pt"  # Ganti dengan path model Anda
    model = load_model(model_path)
    
    # Sidebar: Pilih mode (gambar atau video)
    mode = st.sidebar.radio("Pilih mode deteksi:", ("Real-time Kamera", "Unggah Gambar"))

    # Sidebar: Atur confidence threshold
    confidence_threshold = st.sidebar.slider("Confidence Threshold:", 0.1, 1.0, 0.5, 0.05)
    
    if mode == "Unggah Gambar":
        st.subheader("📤 *Unggah Gambar untuk Deteksi Objek*")
        uploaded_file = st.file_uploader("Pilih file gambar:", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)

            # Run YOLO deteksi pada gambar
            results = model.predict(image_np, imgsz=640)
            image_np, detected_objects = display_results(image_np, results[0], confidence_threshold)
            
            # Tampilkan gambar dengan hasil deteksi
            st.image(image_np, caption="Hasil Deteksi", use_column_width=True)
            
            # Tampilkan info deteksi
            if detected_objects:
                object_counts = Counter(detected_objects)
                st.markdown("### *📊 Objek Terdeteksi:*")
                for obj, count in object_counts.items():
                    st.write(f"- *{obj}*: {count}")
            else:
                st.write("❌ Tidak ada objek terdeteksi.")
    
    elif mode == "Real-time Kamera":
        st.subheader("📹 *Deteksi Objek Real-time*")
        run_detection = st.sidebar.checkbox("Mulai Deteksi", key="detection_control")

        if run_detection:
            cap = cv2.VideoCapture(0)  # Buka kamera
            st_frame = st.empty()  # Placeholder untuk video
            st_detection_info = st.empty()  # Placeholder untuk info deteksi
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("❌ Gagal menangkap gambar dari kamera.")
                    break
                
                # Konversi frame ke RGB dan lakukan deteksi
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model.predict(frame, imgsz=640)
                frame, detected_objects = display_results(frame, results[0], confidence_threshold)
                
                # Tampilkan hasil video dan info deteksi
                st_frame.image(frame, channels="RGB", use_column_width=True)
                
                if detected_objects:
                    object_counts = Counter(detected_objects)
                    detection_info = "\n".join([f"{obj}: {count}" for obj, count in object_counts.items()])
                else:
                    detection_info = "❌ Tidak ada objek terdeteksi."

                st_detection_info.text(detection_info)

                # Hentikan deteksi jika checkbox dimatikan
                if not st.session_state.detection_control:
                    break
            
            cap.release()
            st.success("🎉 Deteksi objek dihentikan.")
    
    st.sidebar.markdown("---")
    st.sidebar.info("👨‍💻 Dibuat dengan cinta oleh hendri menggunakan Streamlit dan YOLO 11.")

if __name__ == "_main_":
    main()
