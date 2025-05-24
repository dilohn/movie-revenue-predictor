# Movie Revenue Predictor

This project uses a multi-input deep learning model to predict movie revenue based on the movie summary and visual features extracted from its poster. It combines natural language processing (NLP) and computer vision (CV) to generate a single revenue estimate.

## Features

- Processes and embeds text data using bidirectional LSTM layers.
- Extracts visual features from movie posters, including contrast, average RGB values, texture (GLCM), edge detection (Canny), anomaly score, and detected people (YOLOv8).
- Combines textual and numerical features in a single model for prediction.
- Provides both training script and inference module for standalone predictions.

## Files

- `model.py`: Main training pipeline combining text and visual features into a neural network, trains and saves the model and all related components.
- `movie_predictor.py`: Loads the saved model and components for standalone revenue prediction on new movie data.
- `get_image_info.py`: Extracts features from poster images using computer vision techniques like YOLOv8, GLCM, Canny, etc.
- `canny_example.py`: Visualizes the steps in Canny edge detection on a sample image.
- `homogeneity_example.py`: Demonstrates how GLCM and homogeneity are computed and visualized.
- `summary and poster pres.pdf`: A presentation outlining project goals, preprocessing methods, model architecture, and results.
- `madmaxxfury.jpg`, `Mickey 17.jpg`: Sample poster images used for feature extraction and visualization.

## Requirements

Ensure you have the following installed:

- Python 3.x
- TensorFlow / Keras
- OpenCV (`opencv-python`)
- NumPy, Pandas, Matplotlib
- Scikit-image (`scikit-image`)
- NLTK
- Ultralytics YOLOv8 (`ultralytics`)
- Pretrained YOLOv8 model weights (`yolov8x.pt`)

### Sample Output (Mickey 17)
Predicting for a week after it's release
Predicted Revenue: \$78,132,304.00  
Reported Revenue: \$53,300,000.00
