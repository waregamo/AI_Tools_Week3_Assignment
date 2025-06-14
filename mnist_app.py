import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# Set page config
st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")

# Custom CSS for centered layout
st.markdown("""
<style>
    .main {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1rem;
        margin-top: 1rem;
    }
    .prediction-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        width: 100%;
        text-align: center;
    }
    .prediction-text {
        font-size: 2rem;
        font-weight: bold;
        color: #2e86c1;
        margin: 0.5rem 0;
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }
    .image-container img {
        width: 200px;
        height: 200px;
        object-fit: contain;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chart-container {
        width: 100%;
        margin: 1rem auto;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        # First try loading with the modern .keras format
        model = tf.keras.models.load_model("mnist_cnn.keras")
    except:
        try:
            # Fallback to .h5 with compatibility fix
            custom_objects = {'InputLayer': tf.keras.layers.InputLayer}
            model = tf.keras.models.load_model(
                "mnist_cnn.h5",
                custom_objects=custom_objects,
                compile=False
            )
            model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None
    return model

model = load_model()

# Main title
st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9)")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)

if uploaded_file is not None and model is not None:
    try:
        # Process image
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        
        # Display uploaded image (centered, medium size)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(
                image, 
                caption="Uploaded Image", 
                use_container_width=False,
                width=200
            )
        
        # Prepare image for prediction
        img_array = np.array(image)
        img_array = 255 - img_array  # Invert if background is dark
        img_tensor = img_array.reshape(1, 28, 28, 1).astype("float32") / 255.0
        
        # Make prediction
        prediction = model.predict(img_tensor)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Display prediction result
        st.markdown("---")
        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown("### Prediction")
        st.markdown(f'<div class="prediction-text">{predicted_digit}</div>', unsafe_allow_html=True)
        st.markdown(f'Confidence: {confidence*100:.1f}%')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show probability distribution
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Probability Distribution")
        proba_df = pd.DataFrame({
            'Digit': range(10),
            'Probability': prediction[0]
        })
        st.bar_chart(proba_df.set_index('Digit'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
