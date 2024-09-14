import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Function to load the model from a local path
@st.cache_resource
def load_custom_model():
    model_path = 'model/TB_detection_model.keras'  # Ensure this path is correct
    return tf.keras.models.load_model(model_path)

# Load the model
custom_model = load_custom_model()

# Function to preprocess the uploaded image
def preprocess_image(image, target_size=(150, 150)):
    """Resize and normalize the uploaded image to the target size and format required by the model."""
    image = image.convert('RGB')  # Ensure the image is in RGB format
    image = image.resize(target_size)  # Resize the image to match model input size
    image_array = np.array(image)  # Convert image to numpy array
    image_array = image_array / 255.0  # Normalize the pixel values to [0,1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension: (1, 150, 150, 3)
    return image_array

# Function to make predictions using the custom model
def predict(model, image, threshold=0.475):
    """Make a prediction using the model and return 'TB Positive' or 'Normal' based on the threshold."""
    predictions = model.predict(image)

    # Print raw predictions for debugging
    st.write(f'Raw Predictions: {predictions}')  # Print raw prediction array

    # Ensure the model outputs a valid probability value
    if predictions.shape == (1, 1):  # Assuming the model returns a single probability output
        probability = predictions[0][0]
        st.write(f'Prediction Probability: {probability:.4f}')  # Display the probability
        return 'TB Positive' if probability < threshold else 'Normal'
    else:
        return 'Error: Invalid model output shape.'

# Streamlit App Interface
st.title('PulmoScan AI')

# Allow the user to upload an image
uploaded_file = st.file_uploader('Upload a Chest X-ray Image', type=['jpg', 'jpeg', 'png'])

# Add a slider to adjust the threshold value
threshold = st.slider('Select Prediction Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Chest X-ray', use_column_width=True)
    
    # Preprocess the uploaded image
    processed_image = preprocess_image(image)

    # Predict button
    if st.button('Predict'):
        prediction = predict(custom_model, processed_image, threshold)
        st.write(f'Prediction: {prediction}')
