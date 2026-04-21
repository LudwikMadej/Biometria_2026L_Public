import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_pupil(img, x_center, y_center, radius, title="Eye geometry Visualization"):
    """
    Overlays detected pupil geometry on a grayscale image for notebook display.
    
    Visualization specs:
    - Boundary: Neon Green circle (thickness = 1)
    - Center: Red '+' marker (crosshair)
    
    Args:
        img (numpy.ndarray): Input grayscale or RGB image.
        x_center (float): Calculated X-coordinate of the pupil centroid.
        y_center (float): Calculated Y-coordinate of the pupil centroid.
        radius (float): Calculated radius of the pupil.
        title (str): Title displayed above the plot.
    """
    if len(img.shape) == 2:
        output_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
    else:
        output_img = img.copy()


    NEON_GREEN = (57, 255, 20)
    RED = (255, 0, 0)
    
    center = (int(x_center), int(y_center))
    r = int(radius)


    cv2.circle(output_img, center, r, NEON_GREEN, 1)
    cv2.drawMarker(output_img, 
                   center, 
                   RED, 
                   markerType=cv2.MARKER_CROSS, 
                   markerSize=10, 
                   thickness=1)

    # 5. Notebook Rendering
    plt.figure(figsize=(8, 6))
    plt.imshow(output_img)
    plt.title(title)
    plt.axis('off')
    plt.show()
