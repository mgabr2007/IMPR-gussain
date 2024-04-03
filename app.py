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
    image_array = np.array(image)
    blurred_image = cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)
    blurred_image_pil = Image.fromarray(blurred_image)
    return blurred_image_pil

# Define a function for face detection that blurs the background
def detect_faces_and_blur_background(image):
    """
    Blurs the entire image except for detected faces, which remain in black and white.

    Parameters:
    - image: PIL.Image object.

    Returns:
    - PIL.Image object with the background blurred and faces in black and white.
    """
    image_array = np.array(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(image_array, (21, 21), 0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        blurred_image[y:y+h, x:x+w] = cv2.cvtColor(image_array[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    image_with_unblurred_faces = Image.fromarray(blurred_image)
    return image_with_unblurred_faces

# Main function to layout the Streamlit app
def main():
    st.title("Gaussian Blur & Face Detection Image Processing App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        process_option = st.selectbox("Choose an image processing option:", ["Gaussian Blur", "Detect & Preserve Faces"])

        if process_option == "Gaussian Blur":
            kernel_size = st.slider("Select the intensity of the Gaussian Blur (Kernel Size)", 3, 27, 3, step=2)
            if kernel_size % 2 == 0:
                st.error("Kernel size must be an odd number.")
            elif st.button("Apply Gaussian Blur"):
                result_image = apply_gaussian_blur(image, kernel_size)
                st.image(result_image, caption='Blurred Image', use_column_width=True)
                
        elif process_option == "Detect & Preserve Faces":
            if st.button("Detect Faces & Blur Background"):
                result_image = detect_faces_and_blur_background(image)
                st.image(result_image, caption='Background Blurred with Faces in Black & White', use_column_width=True)

if __name__ == "__main__":
    main()
