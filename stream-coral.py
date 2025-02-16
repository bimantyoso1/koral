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

# Streamlit UI
st.title("Coral Guard AI for Classification: Healthy vs. Bleached")
st.image("CoralAi.png", caption="Coral Reef", use_column_width=True)

st.write("""
    CoralGuard-AI is an advanced AI-powered system designed to classify the health of coral reefs, distinguishing between healthy and bleached conditions. This initiative is driven by the critical need to monitor and protect coral reef ecosystems, which are increasingly threatened by environmental stressors such as climate change, pollution, overfishing, and coastal development.

    Coral reef resilience refers to the ability of reef ecosystems to resist disturbances and recover while maintaining biodiversity and ecological functions. Instead of merely preventing change, conservation efforts focus on enhancing the reef’s adaptive capacity, ensuring its survival against extreme weather events, mass bleaching, and other ecological challenges. A resilient coral reef functions like a strong immune system, capable of withstanding and recovering from stressors.

    By adopting a holistic management approach, CoralGuard-AI supports reef conservation by integrating AI-driven classification with ecological monitoring. This ensures that efforts to strengthen reef ecosystems remain data-driven, enabling better decision-making in protecting marine biodiversity.
    
    This application helps classify the condition of coral reefs automatically using **AI**.

    **This project was developed by Bimantyoso Hamdikatama**  
    from Universitas Muhammadiyah Surakarta as an effort to support environmental conservation  
    through Deep Learning technology. This application utilizes a CNN (Convolutional Neural Network) model  
    **to distinguish between healthy and bleached coral reefs based on underwater images.**
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
# Team Contributions

    - **Project Leader**: Bimantyoso Hamdikatama  
    - **AI Model & Machine Learning**: Bimantyoso Hamdikatama, Agus Supriyanto  
    - **Data Processing & Analysis**: Diah Ratnasari, Endang Susanti  
    - **Deployment**: Agung Triatmaja  

    Thank you to everyone who contributed to the development of this project!
""")
