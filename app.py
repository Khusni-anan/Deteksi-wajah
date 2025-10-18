%%writefile app.py
import streamlit as st
import tensorflow as tf
import cv2 # OpenCV
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av # Untuk memproses frame video

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide")

# --- 1. Muat Model ---

# Muat Model Klasifikasi Emosi .h5 Anda
@st.cache_resource
def load_emotion_model():
    try:
        # PENTING: Pastikan nama file ini sama dengan yang Anda upload
        model = tf.keras.models.load_model('model_weights.h5')
        return model
    except Exception as e:
        st.error(f"Error memuat model .h5: {e}")
        st.stop()
        
model = load_emotion_model()

# Muat Model Deteksi Wajah OpenCV
@st.cache_resource
def load_face_cascade():
    try:
        # PENTING: Pastikan nama file ini sama dengan yang Anda upload
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            st.error("Error: Gagal memuat Haar Cascade. Pastikan file 'haarcascade_frontalface_default.xml' ada.")
            st.stop()
        return face_cascade
    except Exception as e:
        st.error(f"Error memuat Haar Cascade: {e}")
        st.stop()

face_cascade = load_face_cascade()

# Definisikan label kelas (URUTAN ALFABET DARI FOLDER 'train')
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# --- 2. Judul Aplikasi ---
st.title("Aplikasi Deteksi Ekspresi Wajah Real-Time 😠😟😊😲")
st.write("Nyalakan kamera Anda dan biarkan model menebak ekspresi Anda secara live!")

# --- 3. Logika Pemrosesan Frame Real-Time ---

# Kita buat kelas untuk memproses video
class EmotionVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = face_cascade
        self.model = model
        self.class_labels = class_labels

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Konversi frame dari AV ke format OpenCV
        img_cv = frame.to_ndarray(format="bgr24")
        
        # Buat gambar grayscale untuk detector
        gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # --- Proses Deteksi Wajah ---
        faces = self.face_cascade.detectMultiScale(gray_img, 
                                                   scaleFactor=1.1, 
                                                   minNeighbors=5, 
                                                   minSize=(30, 30))

        # --- Looping Setiap Wajah yang Ditemukan ---
        for (x, y, w, h) in faces:
            # a. Crop wajah dari gambar grayscale (untuk model .h5)
            roi_gray = gray_img[y:y+h, x:x+w]
            
            # b. Preprocessing (HARUS SAMA PERSIS DENGAN TRAINING)
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            # PENTING: Normalisasi / Rescale
            roi_gray = roi_gray / 255.0
            
            roi_gray = np.expand_dims(roi_gray, axis=0)  # [1, 48, 48]
            roi_gray = np.expand_dims(roi_gray, axis=-1) # [1, 48, 48, 1]

            # c. Lakukan Prediksi Emosi
            prediction = self.model.predict(roi_gray, verbose=0)
            
            # d. Dapatkan label
            confidence = np.max(prediction[0]) * 100
            predicted_class_index = np.argmax(prediction[0])
            predicted_label = self.class_labels[predicted_class_index]
            
            # e. Tulis label dan kotak di gambar ASLI (berwarna / img_cv)
            label_text = f"{predicted_label} ({confidence:.1f}%)"
            
            color_map = {
                'Happy': (0, 255, 0),    # Hijau
                'Angry': (0, 0, 255),    # Merah
                'Sad': (0, 0, 255),      # Merah
                'Surprise': (255, 255, 0), # Kuning
                'Neutral': (255, 0, 0),  # Biru
                'Fear': (128, 0, 128),   # Ungu
                'Disgust': (0, 165, 255) # Oranye
            }
            color = color_map.get(predicted_label, (255, 255, 255)) # Putih default
                
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(img_cv, (x, y-25), (x+w, y), color, -1)
            cv2.putText(img_cv, label_text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2) # Teks Hitam

        # Konversi balik frame dari OpenCV ke AV untuk ditampilkan
        return av.VideoFrame.from_ndarray(img_cv, format="bgr24")

# --- 4. Tampilkan Komponen Webcam di Streamlit ---
st.header("Mode Kamera Live 📸")
webrtc_streamer(
    key="emotion-detector",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=EmotionVideoTransformer,
    media_stream_constraints={"video": True, "audio": False}, 
    async_processing=True 
)

st.write("---") 
st.header("Mode Upload Gambar 🖼️")
st.write("Anda juga masih bisa menguji dengan upload gambar statis di sini.")

# --- 5. (OPSIONAL) Mode Upload Gambar ---
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    image_cv = np.array(image_pil)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB_BGR)
    gray_img = cv2.cvtColor(image_cv, cv2.COLOR_BGR_GRAY)

    faces = face_cascade.detectMultiScale(gray_img, 1.1, 5)

    if len(faces) == 0:
        st.warning("Tidak ada wajah yang terdeteksi di gambar ini.")
    else:
        st.success(f"Berhasil mendeteksi {len(faces)} wajah!")
        for (x, y, w, h) in faces:
            roi_gray = gray_img[y:y+h, x:x+w]
            # Preprocessing yang sama
            roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0 
            roi_gray = np.expand_dims(np.expand_dims(roi_gray, -1), 0)

            prediction = model.predict(roi_gray, verbose=0)
            confidence = np.max(prediction[0]) * 100
            predicted_label = class_labels[np.argmax(prediction[0])]

            label_text = f"{predicted_label} ({confidence:.1f}%)"
            color = (0, 255, 0) if predicted_label == 'Happy' else (0, 0, 255)
            
            cv2.rectangle(image_cv, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(image_cv, (x, y-25), (x+w, y), color, -1)
            cv2.putText(img_cv, label_text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        st.subheader("Hasil Deteksi:")
        final_image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR_RGB)
        st.image(final_image_rgb, use_column_width=True)
