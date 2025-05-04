import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from skimage.feature import graycomatrix, graycoprops

def load_model(model_path="yolov8x.pt"):
    return YOLO(model_path)

def load_data(input_csv):
    df = pd.read_csv(input_csv)
    
    # Initialize new columns in the DataFrame
    df['name'] = None
    df['people'] = None
    df['contrast'] = None
    df['avg_blue'] = None
    df['avg_green'] = None
    df['avg_red'] = None
    df['texture'] = None
    df['edge_detection'] = None
    df['anomaly_score'] = None
    
    return df

def detect_people(image_path, model, confidence_threshold=0.5):
    results = model(image_path)
    
    num_people = 0
    for result in results:
        for box in result.boxes:
            confidence = box.conf[0].item()
            if confidence >= confidence_threshold:
                class_id = int(box.cls[0])
                label = model.names[class_id].lower()
                if label == 'person':
                    num_people += 1
    
    return num_people

def compute_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)

def compute_color_averages(image):
    avg_blue, avg_green, avg_red = cv2.mean(image)[:3]
    return avg_blue, avg_green, avg_red

def compute_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize and quantize the grayscale image for GLCM computation
    small_gray = cv2.resize(gray, (64, 64))
    small_gray = (small_gray / 16).astype(np.uint8) # Quantize to 16 levels
    
    # Compute GLCM with distance 1 and angles [0, 45, 90, 135] degrees
    glcm = graycomatrix(small_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=16, symmetric=True, normed=True)
    
    # Extract homogeneity as a texture measure and average over all directions
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    return homogeneity

def compute_edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
    return edge_density

def compute_anomaly_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    std_dev = np.std(gray)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum() # Normalize histogram
    entropy = -np.sum(hist * np.log2(hist + 1e-7)) # Compute entropy
    
    # Normalize components to a [0, 1] range (thresholds may need adjustment)
    max_std = 80.0 # Assumed maximum standard deviation
    max_entropy = 8.0 # Assumed maximum entropy
    std_component = min(std_dev / max_std, 1.0)
    entropy_component = min(entropy / max_entropy, 1.0)
    
    return (std_component + entropy_component) / 2.0

def extract_image_features(image_path, model, confidence_threshold=0.5):
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} does not exist.")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: {image_path} could not be loaded as an image.")
        return None
    
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    num_people = detect_people(image_path, model, confidence_threshold)
    contrast = compute_contrast(image)
    avg_blue, avg_green, avg_red = compute_color_averages(image)
    texture_value = compute_texture(image)
    edge_density = compute_edge_density(image)
    anomaly_score = compute_anomaly_score(image)
    
    return {
        'name': file_name,
        'people': num_people,
        'contrast': contrast,
        'avg_blue': avg_blue,
        'avg_green': avg_green,
        'avg_red': avg_red,
        'texture': texture_value,
        'edge_detection': edge_density,
        'anomaly_score': anomaly_score
    }

def process_images(df, model, confidence_threshold=0.5):
    for idx, row in df.iterrows():
        local_path = row['local_path']
        features = extract_image_features(local_path, model, confidence_threshold)
        
        if features:
            for key, value in features.items():
                df.at[idx, key] = value
    
    return df

def save_data(df, output_csv):
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")

def main(input_csv="mickey17.csv", output_csv="mickey17_image_data.csv", model_path="yolov8x.pt", confidence_threshold=0.5):
    model = load_model(model_path)
    
    df = load_data(input_csv)
    
    df = process_images(df, model, confidence_threshold)
    
    save_data(df, output_csv)

if __name__ == "__main__":
    main()
