import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import gdown
import os

# URL Google Drive untuk model (Gantilah FILE_ID dengan ID Anda)
file_id = "1akHCW9ZLeyRfYGmSIg8j6vbNM83_Ttsb"
url = f"https://drive.google.com/uc?id={file_id}"

# Path penyimpanan model
model_path = "model_corals.keras"

# Unduh model jika belum ada
if not os.path.exists(model_path):
    st.write("🔄 Mengunduh model, harap tunggu...")
    gdown.download(url, model_path, quiet=False)
    st.write("✅ Model berhasil diunduh!")

# Load model
model = tf.keras.models.load_model(model_path)


# Class labels
class_names = ['Bleached', 'Healthy']

# Display header image
#st.image("https://images.prismic.io/ocean-agency-cms/Z2adyJbqstJ98wZs_RenataCR.jpg?auto=format,compress", caption="Coral Reef", use_column_width=True)

# Streamlit UI
st.title("Coral Reefs Classification: Healthy vs. Bleached")
st.image("https://images.prismic.io/ocean-agency-cms/Z2adyJbqstJ98wZs_RenataCR.jpg?auto=format,compress", caption="Coral Reef", use_column_width=True)

st.write("""
    Terumbu karang adalah ekosistem laut yang sangat penting bagi kehidupan bawah laut. 
    Namun, akibat perubahan iklim dan polusi, banyak terumbu karang mengalami pemutihan (**bleaching**).
    
    Aplikasi ini membantu mengklasifikasikan kondisi terumbu karang secara otomatis menggunakan **AI**.
    
    **Proyek ini dikembangkan oleh Bimantyoso Hamdikatama** 
    dari Universitas Muhammadiyah Surakarta sebagai upaya mendukung pelestarian lingkungan 
    dengan teknologi Deep Learning. Aplikasi ini memanfaatkan model CNN (Convolutional Neural Network) 
    **untuk membedakan antara healthy dan bleached coral reefs berdasarkan citra bawah laut.**
""")

# Upload image
uploaded_file = st.file_uploader("Unggah gambar terumbu karang", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    # Preprocess image
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalisasi
    image_array = np.expand_dims(image_array, axis=0)  # Tambah batch dimension
    
    # Prediction
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    # Show result
    st.write(f"### Prediksi: {predicted_class} (Kepercayaan: {confidence:.2f}%)")

# Team Contributions
st.write("""
## Kontribusi Tim

- **Ketua Proyek**: Bimantyoso Hamdikatama
- **Model AI & Machine Learning**: Bimantyoso Hamdikatama, Agus Supriyanto, Tri Yulianto
- **Pengolahan Data & Analisis**: Diah Ratnasari, Endang Susanti, Agung Triatmaja
- **Anggota** : Nia Rachmawati, Dedi Mizwar, Renata Anita


Terima kasih kepada semua pihak yang telah berkontribusi dalam pengembangan proyek ini!
""")
