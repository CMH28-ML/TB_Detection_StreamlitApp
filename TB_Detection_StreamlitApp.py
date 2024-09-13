import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Function to load the model from a local path
@st.cache(allow_output_mutation=True)
def load_custom_model():
    model_path = 'model\TB_detection_model.keras' 
    return tf.keras.models.load_model(model_path)

# Load the model
custom_model = load_custom_model()

# Function to preprocess the uploaded image
def preprocess_image(image, target_size=(150, 150)):
    image = image.resize(target_size)  # Resize the image to match model input size
    image_array = np.array(image)  # Convert image to numpy array
    image_array = image_array / 255.0  # Normalize the pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to make predictions using the custom model
def predict(model, image):
    predictions = model.predict(image)
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
