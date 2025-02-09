import streamlit as st
import cv2
import numpy as np
from PIL import Image

def pencil_sketch(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_image = cv2.bitwise_not(gray_image)
    blurred_image = cv2.GaussianBlur(inverted_image, (111, 111), 0)
    inverted_blurred_image = cv2.bitwise_not(blurred_image)
    sketch = cv2.divide(gray_image, inverted_blurred_image, scale=255.0)
    sketch = cv2.multiply(sketch, 0.9)
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)
    return sketch

def main():
    st.set_page_config(page_title="Pencil Sketch Converter", layout="wide")
    st.title("🎨 Image to Pencil Sketch Converter")
    st.markdown("Upload multiple images to convert them into pencil sketches!")
    
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        col1, col2 = st.columns(2)
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            sketch = pencil_sketch(image)
            sketch_image = Image.fromarray(sketch)
            
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(sketch_image, caption="Pencil Sketch", use_column_width=True)
                st.download_button(
                    label="Download Sketch",
                    data=cv2.imencode('.png', sketch)[1].tobytes(),
                    file_name=f"sketch_{uploaded_file.name}",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
