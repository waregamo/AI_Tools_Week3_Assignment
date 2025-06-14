import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("mnist_cnn.h5")

st.title("🧠 MNIST Digit Classifier")
st.write("Upload a 28x28 grayscale image of a handwritten digit (0–9)")

uploaded_file = st.file_uploader("Choose a digit image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L").resize((28, 28))  # Grayscale + resize
    st.image(image, caption="Uploaded Image", use_container_width=False)


    img_array = np.array(image)
    img_tensor = img_array.reshape(1, 28, 28, 1).astype("float32") / 255.0

    prediction = model.predict(img_tensor)
    predicted_digit = np.argmax(prediction)

    st.subheader("🧮 Prediction:")
    st.write(f"The model predicts this digit is: **{predicted_digit}**")
