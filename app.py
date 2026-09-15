
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = 224
THRESHOLD = 0.90

MODEL_PATH = "models/chest_xray_pneumonia_efficientnetb0.keras"

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩻",
    layout="wide"
)

st.title("🩻 Pneumonia Detection from Chest X-Rays")
st.write(
    "Deep Learning-based chest X-ray classification using "
    "EfficientNetB0 with Grad-CAM explainability."
)

# -----------------------------
# Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype(np.float32)
    input_tensor = np.expand_dims(img_array, axis=0)

    # Prediction
    probability = float(model.predict(input_tensor, verbose=0)[0][0])

    if probability >= THRESHOLD:
        predicted_class = "PNEUMONIA"
        confidence = probability
    else:
        predicted_class = "NORMAL"
        confidence = 1 - probability

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    with col2:
        st.metric("Prediction", predicted_class)
        st.metric("Confidence", f"{confidence:.2%}")

        if predicted_class == "PNEUMONIA":
            st.warning(
                "The model predicts PNEUMONIA. "
                "This is an AI prediction and not a medical diagnosis."
            )
        else:
            st.success(
                "The model predicts NORMAL. "
                "This is an AI prediction and not a medical diagnosis."
            )

    # -----------------------------
    # Grad-CAM
    # -----------------------------
    base_model = model.get_layer("efficientnetb0")

    gap_layer = model.layers[2]
    dropout1_layer = model.layers[3]
    dense_layer = model.layers[4]
    dropout2_layer = model.layers[5]
    output_layer = model.layers[6]

    with tf.GradientTape() as tape:

        conv_outputs = base_model(
            input_tensor,
            training=False
        )

        tape.watch(conv_outputs)

        x = gap_layer(conv_outputs)
        x = dropout1_layer(x, training=False)
        x = dense_layer(x)
        x = dropout2_layer(x, training=False)
        prediction = output_layer(x)

    grads = tape.gradient(
        prediction,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    heatmap = heatmap.numpy()

    # Convert image
    original = np.array(image)

    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    st.subheader("Grad-CAM Explainability")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            heatmap,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

    with col2:
        st.image(
            overlay,
            caption="Grad-CAM Overlay",
            use_container_width=True
        )

    st.caption(
        "Highlighted regions indicate areas that contributed "
        "more strongly to the model's prediction."
    )
