import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Page configuration
st.set_page_config(
    page_title="MNIST Classifier",
    page_icon="🔢",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .st-cn { background-color: black !important; }
    canvas { cursor: crosshair !important; }
    .uploaded-image { max-width: 280px; max-height: 280px; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px 4px 0 0;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        color: white;
        text-align: center;
        padding: 10px;
        border-top: 1px solid #2d3741;
        font-size: 0.8em;
    }
    .model-info {
        background-color: #1a1a1a;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #1a1a1a;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Model loading
@st.cache_resource
def load_model():
    model_paths = [
        "models/mnist_cnn.keras",
        "models/mnist_cnn.h5"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                st.success(f"Successfully loaded model from {path}")
                return model
            except Exception as e:
                st.warning(f"Failed to load {path}: {str(e)}")
                continue
    
    st.error("""
    No valid model found. Please verify:
    1. The model files exist in models/ directory
    2. Files are named correctly (mnist_cnn.keras/mnist_cnn.h5)
    """)
    st.write("Current directory:", os.getcwd())
    st.write("Models directory contents:", os.listdir("models"))
    st.stop()

def preprocess_image(image):
    """Preprocess image for MNIST model"""
    img = image.convert('L').resize((28, 28))
    img_array = 1 - (np.array(img) / 255.0)  # Invert and normalize
    return img_array.reshape(1, 28, 28, 1)

def show_prediction_results(img, prediction):
    """Display prediction results"""
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Input Image", width=150)
    with col2:
        st.success(f"Predicted Digit: **{pred_class}**")
        st.metric("Confidence", f"{confidence:.1%}")
        
        # Show probabilities
        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.bar(range(10), prediction[0], color='skyblue')
        bars[pred_class].set_color('orange')
        ax.set_xticks(range(10))
        ax.set_title("Class Probabilities")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Digits")
        ax.set_ylabel("Probability")
        st.pyplot(fig)

def show_model_info():
    """Display detailed model information"""
    with st.expander("ℹ️ Model Information", expanded=True):
        st.markdown("""
        <div class="model-info">
            <h4 style="margin-top:0;">CNN Architecture</h4>
            <div class="metric-box">
                <strong>Layers:</strong><br>
                • Conv2D (32 filters, 3x3, ReLU)<br>
                • MaxPooling2D (2x2)<br>
                • Conv2D (64 filters, 3x3, ReLU)<br>
                • MaxPooling2D (2x2)<br>
                • Flatten<br>
                • Dropout (0.25)<br>
                • Dense (128 units, ReLU)<br>
                • Dense (10 units, Softmax)
            </div>
            <div class="metric-box">
                <strong>Training Parameters:</strong><br>
                • Optimizer: Adam<br>
                • Loss: Sparse Categorical Crossentropy<br>
                • Batch Size: 64<br>
                • Epochs: 10
            </div>
            <div class="metric-box">
                <strong>Performance Metrics:</strong><br>
                • Test Accuracy: 99.38%<br>
                • Validation Accuracy: 99.12%<br>
                • Training Time: ~2 minutes (on CPU)
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_ethical_footer():
    """Display ethical considerations footer"""
    st.markdown("""
    <div class="footer">
        <strong>Ethical Considerations:</strong> This AI model is trained on the MNIST dataset which contains handwritten digits. 
        While generally accurate, it may make errors with ambiguous or poorly written digits. 
        Not recommended for critical applications without human verification.
    </div>
    """, unsafe_allow_html=True)

def main():
    model = load_model()
    st.title("MNIST Digit Classifier")
    
    # Show model information
    show_model_info()
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["✏️ Draw Digit", "📁 Upload Image"])
    
    with tab1:
        st.header("Draw a Digit")
        st.write("Draw a digit (0-9) in the canvas below:")
        
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)",
            stroke_width=15,
            stroke_color="rgba(255, 255, 255, 1)",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas"
        )
        
        if st.button("Predict Drawing", type="primary"):
            if canvas_result.image_data is not None:
                try:
                    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                    img_array = preprocess_image(img)
                    prediction = model.predict(img_array)
                    show_prediction_results(img, prediction)
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
            else:
                st.warning("Please draw a digit first!")
    
    with tab2:
        st.header("Upload an Image")
        st.write("Upload an image of a digit (0-9):")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", width=150)
                
                if st.button("Predict Uploaded Image", type="primary"):
                    img_array = preprocess_image(img)
                    prediction = model.predict(img_array)
                    show_prediction_results(img, prediction)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    # Show ethical footer
    show_ethical_footer()

if __name__ == "__main__":
    main()