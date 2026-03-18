import streamlit as st
import numpy as np
import tensorflow as tf
from keras.utils import img_to_array
from PIL import Image
from streamlit_cropper import st_cropper

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("models/cable_classifier_mobilenetv2")

model = load_my_model()

class_names = [
    'HDMI Female', 
    'HDMI Male', 
    'USB-A Female', 
    'USB-A Male', 
    'USB-C Female', 
    'USB-C Male'
]

connector_info = {
    'HDMI Female': {
        'description': 'This is an HDMI female port, typically found on TVs, monitors, projectors, and laptops. It receives an HDMI male connector.',
        'usage': 'Plug an HDMI male cable into this port to transmit audio and video signals.',
        'compatible_with': ['HDMI Male']
    },
    'HDMI Male': {
        'description': 'This is an HDMI male connector, found at the end of HDMI cables. It carries digital audio and video signals.',
        'usage': 'Insert this connector into an HDMI female port on a TV, monitor, or projector.',
        'compatible_with': ['HDMI Female']
    },
    'USB-A Female': {
        'description': 'This is a USB-A female port, commonly found on computers, laptops, chargers, and hubs. It receives a USB-A male connector.',
        'usage': 'Plug a USB-A male cable or device into this port.',
        'compatible_with': ['USB-A Male']
    },
    'USB-A Male': {
        'description': 'This is a USB-A male connector, the most common USB plug found on cables, flash drives, and peripherals.',
        'usage': 'Insert this connector into a USB-A female port on a computer, charger, or hub.',
        'compatible_with': ['USB-A Female']
    },
    'USB-C Female': {
        'description': 'This is a USB-C female port, found on modern phones, laptops, tablets, and accessories.',
        'usage': 'Plug a USB-C male cable into this port for charging, data transfer, or video output.',
        'compatible_with': ['USB-C Male']
    },
    'USB-C Male': {
        'description': 'This is a USB-C male connector, a modern reversible plug found on cables and adapters.',
        'usage': 'Insert this connector into a USB-C female port. It works in either orientation.',
        'compatible_with': ['USB-C Female']
    }
}

st.title("🔌 Cable Connector Classifier")
st.write("Upload an image or take a photo of a cable connector to identify it.")

input_method = st.radio("Choose input method:", ["📁 Upload from gallery", "📸 Take a photo"])

if input_method == "📁 Upload from gallery":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
else:
    uploaded_file = st.camera_input("Take a photo")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, caption="Original image", use_column_width=True)

    use_crop = st.checkbox("Crop image before prediction", value=False)

    image_to_predict = image

    if use_crop:
        st.subheader("Crop Image")
        cropped_img = st_cropper(
            image,
            realtime_update=True,
            box_color="#0000FF",
            aspect_ratio=None
        )

        st.subheader("Cropped Image")
        st.image(cropped_img, caption="Cropped image", use_column_width=True)

        image_to_predict = cropped_img

    if st.button("Predict"):
        # Preprocess selected image
        img = image_to_predict.resize((224, 224))
        img_array = np.expand_dims(img_to_array(img), axis=0)

        # Predict
        prediction = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction)) * 100

        # Results
        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.1f}%")

        # Show only one bar
        st.subheader("Class Probability")
        st.progress(float(np.max(prediction)), text=f"{predicted_class}: {confidence:.1f}%")

        # Connector details
        info = connector_info[predicted_class]

        st.subheader("📖 Description")
        st.write(info["description"])

        st.subheader("🛠️ How to Use")
        st.write(info["usage"])

        st.subheader("🔗 Compatible With")
        for item in info["compatible_with"]:
            st.write(f"- {item}")