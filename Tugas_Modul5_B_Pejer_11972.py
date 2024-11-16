import streamlit as st
import numpy as np
import pickle
from PIL import Image
import os

# Path relatif untuk model
model_path = "best_model.pkl"

# Load model
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        def preprocess_image(image):
            image = image.resize((28, 28))  # Ubah ukuran menjadi 28x28 piksel
            image = image.convert('L')  # Ubah menjadi grayscale
            image_array = np.array(image) / 255.0  # Normalisasi
            image_array = image_array.reshape(1, -1)  # Flatten ke bentuk 1D array
            return image_array

        # Judul aplikasi
        st.title("Fashion MNIST Image Classifier")
        st.write("Unggah beberapa gambar item fashion (misalnya sepatu, tas, baju), dan model akan memprediksi kelas masing-masing.")

        # File uploader
        uploaded_files = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        # Sidebar dengan tombol prediksi
        with st.sidebar:
            st.write("### Navigation")
            predict_button = st.button("Predict")

        # Proses prediksi
        if uploaded_files and predict_button:
            st.write("### Hasil Prediksi")
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                processed_image = preprocess_image(image)
                predictions = model.predict_proba(processed_image)
                predicted_class = np.argmax(predictions)
                confidence = np.max(predictions) * 100

                # Tampilkan hasil prediksi
                st.write(f"**Nama File:** {uploaded_file.name}")
                st.write(f"**Kelas Prediksi:** {class_names[predicted_class]}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.write("---")

        # Tampilkan gambar yang diunggah
        if uploaded_files:
            st.write("### Gambar yang Diupload")
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Gambar: {uploaded_file.name}", use_column_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.error("File model tidak ditemukan. Pastikan model sudah diunggah di folder 'models'.")
