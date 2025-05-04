import numpy as np
import os
import pandas as pd
import pickle
import re
import warnings
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Make the code stop being annoying
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR') 

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


class MovieRevenuePredictor:
    """
    A class to load a pre-trained movie revenue prediction model and make predictions.
    This allows predictions without retraining the model each time.
    """
    
    def __init__(self, models_dir='models'):
        # NLTK resources are downloaded at the module level
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory '{models_dir}' not found")

        model_path = os.path.join(models_dir, 'movie_revenue_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = load_model(model_path)
        
        # Load the tokenizer
        tokenizer_path = os.path.join(models_dir, 'tokenizer.pickle')
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        # Load the numerical scaler
        numerical_scaler_path = os.path.join(models_dir, 'numerical_scaler.pickle')
        if not os.path.exists(numerical_scaler_path):
            raise FileNotFoundError(f"Numerical scaler file not found at {numerical_scaler_path}")
        with open(numerical_scaler_path, 'rb') as handle:
            self.numerical_scaler = pickle.load(handle)
        
        # Load the y scaler
        y_scaler_path = os.path.join(models_dir, 'y_scaler.pickle')
        if not os.path.exists(y_scaler_path):
            raise FileNotFoundError(f"Y scaler file not found at {y_scaler_path}")
        with open(y_scaler_path, 'rb') as handle:
            self.y_scaler = pickle.load(handle)
        
        # Load the config
        config_path = os.path.join(models_dir, 'config.pickle')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path, 'rb') as handle:
            config = pickle.load(handle)
            self.max_sequence_length = config['max_sequence_length']
            self.numerical_features = config['numerical_features']
    
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        words = text.split()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    
    def predict(self, overview, contrast, avg_blue, avg_green, avg_red, texture, edge_detection, anomaly_score, people):
        # Preprocess the text input
        processed_input = self.preprocess_text(overview)
        
        # Convert to sequence
        input_seq = self.tokenizer.texts_to_sequences([processed_input])
        
        # Pad sequence
        input_padded = pad_sequences(input_seq, maxlen=self.max_sequence_length, padding='post')
        
        # Prepare numerical features
        numerical_input = np.array([[contrast, avg_blue, avg_green, avg_red, texture, edge_detection, anomaly_score, people]])
        numerical_input_scaled = self.numerical_scaler.transform(numerical_input)
        
        pred_scaled = self.model.predict([input_padded, numerical_input_scaled], verbose=0)
        prediction = self.y_scaler.inverse_transform(pred_scaled).flatten()[0]
        
        return prediction


def main():
    try:
        predictor = MovieRevenuePredictor()
        
        csv_path = 'mickey17_image_data.csv'
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            title = row['title']
            overview = row['overview']
            contrast = float(row['contrast'])
            avg_blue = float(row['avg_blue'])
            avg_green = float(row['avg_green'])
            avg_red = float(row['avg_red'])
            texture = float(row['texture'])
            edge_detection = float(row['edge_detection'])
            anomaly_score = float(row['anomaly_score'])
            people = float(row['people'])
            
            predicted_revenue = predictor.predict(
                overview, contrast, avg_blue, avg_green, avg_red, texture, 
                edge_detection, anomaly_score, people
            )
            
            print(f"${predicted_revenue:,.2f}")
            
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
