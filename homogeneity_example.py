import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

def load_and_preprocess_image(image_path, quantization_levels=8, resize_dim=(64, 64)):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the grayscale image to 64x64
    gray_resized = cv2.resize(gray, resize_dim)

    # Create a separate quantized version AFTER resizing
    quantized_gray = (gray_resized / (256 // quantization_levels)).astype(np.uint8)

    return image, gray_resized, quantized_gray

def compute_glcm_features(quantized_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16):
    glcm = graycomatrix(quantized_gray, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    
    # Extract homogeneity feature
    homogeneity = graycoprops(glcm, 'homogeneity')
    mean_homogeneity = homogeneity.mean()

    return glcm, homogeneity, mean_homogeneity

def display_image(title, img, cmap=None):
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()

def display_glcm(glcm, directions):
    for i, direction in enumerate(directions):
        plt.figure(figsize=(6,6))
        plt.imshow(glcm[:, :, 0, i], cmap='viridis', interpolation='nearest')
        plt.title(f"GLCM Matrix - {direction}")
        plt.xlabel("Gray Level j")
        plt.ylabel("Gray Level i")
        plt.show()

def display_homogeneity(homogeneity, directions):
    plt.figure(figsize=(8,6))
    plt.bar(directions, homogeneity[0], color='gold')
    plt.title("Homogeneity Across Directions")
    plt.ylabel("Homogeneity")
    plt.xticks(rotation=45)
    plt.show()

def main(image_path):
    directions = ["0° (Horizontal)", "45° (Diagonal)", "90° (Vertical)", "135° (Diagonal)"]
    
    # Load image and preprocess
    image, gray_resized, quantized_gray = load_and_preprocess_image(image_path)

    # Compute GLCM and homogeneity
    glcm, homogeneity, mean_homogeneity = compute_glcm_features(quantized_gray)

    # Display images separately
    display_image("Original Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    display_image("Grayscale Image (Resized to 64x64)", gray_resized, cmap='gray')
    display_image(f"Quantized Image (8 Levels Demo - 16 Levels Application)", quantized_gray, cmap='gray')

    # Display GLCM matrices separately
    display_glcm(glcm, directions)

    # Display homogeneity chart
    display_homogeneity(homogeneity, directions)

image_path = "madmaxxfury.jpg" 
main(image_path)
