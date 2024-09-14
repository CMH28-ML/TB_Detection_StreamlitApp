import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Function to load the model from a local path
@st.cache_resource
def load_custom_model():
    model_path = 'model/TB_detection_model.keras'  # Make sure this path is correct
    return tf.keras.models.load_model(model_path)

# Load the model
custom_model = load_custom_model()

# Function to preprocess the uploaded image
def preprocess_image(image, target_size=(150, 150)):
    """Resize and normalize the uploaded image to the target size and format required by the model"""
    image = image.convert('RGB')  # Ensure the image is in RGB format
    image = image.resize(target_size)  # Resize the image to match model input size
    image_array = np.array(image)  # Convert image to numpy array
    image_array = image_array / 255.0  # Normalize the pixel values to [0,1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension: (1, 150, 150, 3)
    return image_array

# Function to make predictions using the custom model
def predict(model, image):
    """Make a prediction using the model and return 'TB Positive' or 'Normal'"""
    predictions = model.predict(image)
    # Assuming model outputs a single probability: (1, 1)
    return 'TB Positive' if predictions[0][0] > 0.5 else 'Normal'

# Streamlit App Interface
st.title('PulmoScan AI')

# Allow the user to upload an image
uploaded_file = st.file_uploader('Upload a Chest X-ray Image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Chest X-ray', use_column_width=True)
    
    # Preprocess the uploaded image
    processed_image = preprocess_image(image)

    # Predict button
    if st.button('Predict'):
        prediction = predict(custom_model, processed_image)
        st.write(f'Prediction: {prediction}')
