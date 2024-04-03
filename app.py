# Import necessary libraries
import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Define a function to apply Gaussian Blur to an image
def apply_gaussian_blur(image, kernel_size):
    """
    Apply Gaussian Blur to an image.

    Parameters:
    - image: PIL.Image object.
    - kernel_size: int, size of the kernel, must be odd.

    Returns:
    - PIL.Image object after applying Gaussian Blur.
    """
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Apply Gaussian Blur using OpenCV
    blurred_image = cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)
    
    # Convert numpy array back to PIL image
    blurred_image_pil = Image.fromarray(blurred_image)
    
    return blurred_image_pil

# Main function to layout the Streamlit app
def main():
    st.title("Gaussian Blur Image Processing App")

    # Create an image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Slider to select the intensity of the Gaussian Blur
        kernel_size = st.slider("Select the intensity of the Gaussian Blur (Kernel Size)", 3, 27, 3, step=2)
        
        # Ensure the kernel size is odd, as required
        if kernel_size % 2 == 0:
            st.error("Kernel size must be an odd number.")
        else:
            # Button to apply Gaussian Blur
            if st.button("Apply Gaussian Blur"):
                # Apply Gaussian Blur to the uploaded image
                result_image = apply_gaussian_blur(image, kernel_size)
                
                # Display the blurred image
                st.image(result_image, caption='Blurred Image', use_column_width=True)

# Define a function for face detection
import face_recognition

def detect_faces(image):
    """
    Detect faces in an image using the face_recognition library.

    Parameters:
    - image: PIL.Image object.

    Returns:
    - PIL.Image object with rectangles drawn around detected faces.
    """
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Use the face_recognition library to find face locations
    face_locations = face_recognition.face_locations(image_array)
    
    # Use PIL to draw rectangles around each face
    draw = ImageDraw.Draw(image)
    for top, right, bottom, left in face_locations:
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=2)
    
    return image


# Main function to layout the Streamlit app
def main():
    st.title("Gaussian Blur & Face Detection Image Processing App")

    # Create an image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Choose processing option
        process_option = st.selectbox("Choose an image processing option:", ["Gaussian Blur", "Face Detection"])
        
        if process_option == "Gaussian Blur":
            kernel_size = st.slider("Select the intensity of the Gaussian Blur (Kernel Size)", 3, 27, 3, step=2)
            if kernel_size % 2 == 0:
                st.error("Kernel size must be an odd number.")
            elif st.button("Apply Gaussian Blur"):
                result_image = apply_gaussian_blur(image, kernel_size)
                st.image(result_image, caption='Blurred Image', use_column_width=True)
                
        elif process_option == "Face Detection":
            if st.button("Detect Faces"):
                result_image = detect_faces(image)
                st.image(result_image, caption='Image with Detected Faces', use_column_width=True)

if __name__ == "__main__":
     main()
