import cv2
import numpy as np

def visualize_canny_steps(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Resize image to have a width of 500 pixels while maintaining aspect ratio
    height, width = image.shape[:2]
    new_width = 500
    new_height = int((new_width / width) * height) 
    image = cv2.resize(image, (new_width, new_height))

    # Show Original Image
    cv2.imshow("1 - Original Image", image)

    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("2 - Grayscale Image", gray)

    # Apply Gaussian Blur to Remove Noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    cv2.imshow("3 - Gaussian Blurred", blurred)

    # Compute Gradient Magnitude using Sobel Operator
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3) # Gradient in X direction
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3) # Gradient in Y direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2) # Compute Gradient Magnitude
    magnitude = np.uint8(255 * magnitude / np.max(magnitude)) # Normalize for visualization
    cv2.imshow("4 - Gradient Magnitude (Sobel)", magnitude)

    # Apply Canny Edge Detection
    edges = cv2.Canny(gray, 100, 200) # Canny Edge Detection with thresholds 100 & 200
    cv2.imshow("5 - Final Canny Edge Detection", edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = r"Mickey 17.jpg"
visualize_canny_steps(image_path)
