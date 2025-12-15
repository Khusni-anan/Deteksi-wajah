import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import joblib
import json
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Hybrid Emotion Detector", layout="wide")

# --- 1. Muat Resources (Model & Label) ---

@st.cache_resource
def load_components():
    try:
        # 1. Muat Feature Extractor (CNN)
        feature_extractor = tf.keras.models.load_model('emotion_feature_extractor.h5')
        
        # 2. Muat Classifier (Random Forest)
        rf_model = joblib.load('emotion_rf_model.pkl')
        
        # 3. Muat Label Emosi
        with open('emotion_labels.json', 'r') as f:
            label_map = json.load(f)
        # Mengubah key string angka "0" menjadi integer 0
        labels = {int(k): v for k, v in label_map.items()}
        
        # 4. Muat Face Detector (Haar Cascade)
        # Pastikan Anda mendownload file xml ini dari internet jika belum ada
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        return feature_extractor, rf_model, labels, face_cascade
    except Exception as e:
        st.error(f"Gagal memuat file model/label. Pastikan file .h5, .pkl, dan .json ada di folder yang sama.\nError: {e}")
        return None, None, None, None

feature_extractor, rf_model, emotion_labels, face_cascade = load_components()

# --- 2. Judul Aplikasi ---
st.title("Deteksi Emosi Wajah (Hybrid CNN + Random Forest) 😠😊😲")
st.markdown("""
Aplikasi ini menggunakan **CNN** untuk ekstraksi fitur dan **Random Forest** untuk klasifikasi.
Dataset: CK+ (Extended Cohn-Kanade).
""")

# --- 3. Logika Pemrosesan Video ---
class HybridEmotionProcessor(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Konversi frame ke format OpenCV
        img = frame.to_ndarray(format="bgr24")
        
        # Jika model gagal dimuat, kembalikan frame asli
        if feature_extractor is None or rf_model is None:
            return frame

        # Konversi ke Grayscale untuk deteksi wajah
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # a. Crop area wajah
            face_roi = gray[y:y+h, x:x+w]
            
            # b. Preprocessing (SESUAIKAN DENGAN NOTEBOOK)
            # Resize ke 96x96 (bukan 48x48)
            resized = cv2.resize(face_roi, (96, 96))
            
            # Normalisasi [0, 1]
            normalized = resized / 255.0
            
            # Reshape ke (1, 96, 96, 1) agar sesuai input CNN
            input_data = np.reshape(normalized, (1, 96, 96, 1))
            
            try:
                # c. Langkah 1: Ekstraksi Fitur dengan CNN
                features = feature_extractor.predict(input_data, verbose=0)
                
                # d. Langkah 2: Prediksi Kelas dengan Random Forest
                # RF outputnya langsung label angka (misal: 0, 1, 4)
                prediction_idx = rf_model.predict(features)[0]
                
                # Mapping ke nama emosi
                prediction_text = emotion_labels.get(prediction_idx, "Unknown")
                
                # Warna kotak (Hijau untuk Happy, Merah untuk lainnya)
                color = (0, 255, 0) if prediction_text.lower() == 'happy' else (0, 0, 255)
                
                # Gambar kotak dan teks
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, prediction_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except Exception as e:
                print(f"Prediction Error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. Tampilan WebRTC ---
if feature_extractor is not None:
    st.write("### Mode Live Webcam")
    webrtc_streamer(
        key="hybrid-emotion",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=HybridEmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
else:
    st.warning("Model belum siap. Silakan periksa file model Anda.")

# --- 5. Mode Upload Gambar ---
st.write("---")
st.write("### Mode Upload Gambar")
uploaded_file = st.file_uploader("Upload gambar wajah", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None and feature_extractor is not None:
    # Baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, 1)
    
    # Proses sama seperti video
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    st.image(img_cv, channels="BGR", caption="Gambar Asli", width=400)
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            resized = cv2.resize(face_roi, (96, 96))
            normalized = resized / 255.0
            input_data = np.reshape(normalized, (1, 96, 96, 1))
            
            # Prediksi Hybrid
            features = feature_extractor.predict(input_data, verbose=0)
            prediction_idx = rf_model.predict(features)[0]
            prediction_text = emotion_labels.get(prediction_idx, "Unknown")
            
            # Gambar hasil
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_cv, prediction_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        st.image(img_cv, channels="BGR", caption="Hasil Deteksi", width=400)
        st.success(f"Terdeteksi emosi: **{prediction_text}**")
    else:
        st.warning("Wajah tidak terdeteksi.")
