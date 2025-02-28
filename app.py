import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# ✅ Function to find the last convolutional layer
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError("No convolutional layer found in the model.")

# ✅ Grad-CAM Implementation
def grad_cam(image, model):
    last_conv_layer_name = find_last_conv_layer(model)
    img_size = (224, 224)

    # Preprocess image
    img = image.resize(img_size)
    
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Define a model for Grad-CAM
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        tape.watch(conv_outputs)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    # Apply Grad-CAM heatmap
    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Convert heatmap to color
    heatmap = cv2.resize(heatmap, img_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    img = np.array(image)
    img = cv2.resize(img, img_size)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return superimposed_img, class_idx

# ✅ Streamlit UI
st.title("🔍 Grad-CAM Visualization")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load pre-trained model (Example: EfficientNet)
    model = tf.keras.applications.EfficientNetB0(weights="imagenet")

    # Run Grad-CAM
    processed_image, predicted_class = grad_cam(image, model)

    # Display results
    st.image(processed_image, caption=f"Grad-CAM Heatmap (Class: {predicted_class})", use_column_width=True)
    st.success(f"Predicted Class Index: {predicted_class}")
