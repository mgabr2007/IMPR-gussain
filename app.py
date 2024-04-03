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
def detect_faces(image):
    """
    Detect faces in an image using OpenCV's Haar cascade classifier.

    Parameters:
    - image: PIL.Image object.

    Returns:
    - PIL.Image object with rectangles drawn around detected faces.
    """
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Convert to grayscale for face detection
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Load Haar cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Draw rectangles around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(image_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Convert numpy array back to PIL image
    image_with_faces = Image.fromarray(image_array)
    
    return image_with_faces

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
