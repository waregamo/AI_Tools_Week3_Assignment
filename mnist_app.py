import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# Custom CSS
st.markdown("""
<style>
    .image-container img {
        width: 200px;
        height: 200px;
        object-fit: contain;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-container {
        margin-top: 2rem;
    }
    .prediction-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-text {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
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
        model = tf.keras.models.load_model("mnist_cnn.keras")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

model = load_model()

# Main app
st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9)")

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["png", "jpg", "jpeg"],
    key="file_uploader"
)

if uploaded_file is not None:
    try:
        # Process image
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        
        # Display original and processed images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", width=200)
        
        # Preprocess for model
        image = image.resize((28, 28))
        img_array = np.array(image)
        img_array = img_array.reshape(1, 28, 28, 1).astype("float32") / 255.0
        
        # Invert if background is dark (optional)
        if np.mean(img_array) < 0.5:
            img_array = 1 - img_array
        
        with col2:
            st.image(img_array[0,:,:,0], caption="Processed Image", width=200)
        
        # Make prediction
        if model is not None:
            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Display results
            st.markdown("---")
            st.markdown("### Prediction Results")
            
            col_pred, col_conf = st.columns(2)
            with col_pred:
                st.markdown("**Predicted Digit**")
                st.markdown(f'<div class="prediction-text">{predicted_digit}</div>', 
                           unsafe_allow_html=True)
            
            with col_conf:
                st.markdown("**Confidence**")
                st.progress(float(confidence))
                st.markdown(f"{confidence*100:.1f}%")
            
            # Probability chart
            st.markdown("### Probability Distribution")
            proba_df = pd.DataFrame({
                'Digit': range(10),
                'Probability': prediction[0]
            })
            st.bar_chart(proba_df.set_index('Digit'))
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")