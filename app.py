import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('mymodel.h5')

# Define the labels
labels = ['Buildings', 'Forest', 'Sea']

# Define the image classification function
def classify_image(image):
    # Preprocess the image
    image = np.array(image)
    image = tf.image.resize(image, (128, 128))  # Resize the image to match the input size of the model
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    image = tf.keras.applications.resnet50.preprocess_input(image)

    # Predict the class
    predictions = model.predict(image).flatten()

    # Get the predicted class label
    predicted_class = labels[np.argmax(predictions)]

    return predicted_class

# Define the Gradio interface
image_input = gr.Image(type="pil", label="Upload Image")
label_output = gr.Label()

# Create the Gradio interface
interface = gr.Interface(fn=classify_image, inputs=image_input, outputs=label_output, title="Image Classifier")

# Launch the interface
interface.launch()