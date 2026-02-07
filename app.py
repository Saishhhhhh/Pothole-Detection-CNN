import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# Setup
st.set_page_config(page_title="Pothole Detection App", layout="centered")
st.title("Pothole Detection using ResNet50")
st.write("Upload an image to check for potholes.")

# Model Loading with Cache
@st.cache_resource
def load_trained_model():
    model_path = './model/pothole_detection_model.h5'
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

if model:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # Convert to RGB early to handle PNGs with RGBA
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            # Preprocessing
            img = image.resize((224, 224))
            img_array = img_to_array(img)
                
            img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))
            
            # Prediction
            with st.spinner('Analyzing...'):
                prediction = model.predict(img_preprocessed)[0][0]
                
            # Logic: If output neuron is > 0.5 it's class 1 (Pothole), else class 0 (Normal)
            # Assuming training data flow: 0=Normal, 1=Pothole (based on alphabetical order defaults in flow_from_directory)
            # But let's double check logic from user snippet: `status = "POTHOLE" if prediction > 0.5 else "NORMAL"`
            
            status = "POTHOLE DETECTED" if prediction > 0.5 else "NORMAL ROAD"
            confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
            
            # Display Result
            if prediction > 0.5:
                st.error(f"**Result:** {status}")
            else:
                st.success(f"**Result:** {status}")
                
            st.write(f"**Confidence:** {confidence:.2f}%")
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
else:
    st.warning("Model failed to load. Please check the model path.")
