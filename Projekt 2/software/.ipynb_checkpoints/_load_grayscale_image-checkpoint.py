import cv2
import os

def load_grayscale_image(file_path):
    """
    Loads an iris image from a .bmp format.
    
    Args:
        file_path (str): Full path to the .bmp file.
        
    Returns:
        numpy.ndarray: The loaded image in grayscale (or None in case of an error).
    """
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at: {file_path}")
        return None

    # Load the image in grayscale mode
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not decode the image.")
        return None

    return image