import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import os

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

df = pd.read_csv('movies_image_data.csv')

numerical_features = ['contrast', 'avg_blue', 'avg_green', 'avg_red', 'texture', 'edge_detection', 'anomaly_score', 'people']

def preprocess_text(text):
    text = text.lower()
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    words = text.split()
    
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

df['processed_overview'] = df['overview'].apply(preprocess_text)

X_text = df['processed_overview'].values
X_numerical = df[numerical_features].values
y = df['revenue'].values

X_text_train, X_text_test, X_numerical_train, X_numerical_test, y_train, y_test = train_test_split(
    X_text, X_numerical, y, test_size=0.2, random_state=42
)

# Scale the target variable
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

# Scale the numerical features
numerical_scaler = MinMaxScaler()
X_numerical_train_scaled = numerical_scaler.fit_transform(X_numerical_train)
X_numerical_test_scaled = numerical_scaler.transform(X_numerical_test)

# Tokenize the text
max_words = 10000 # Maximum number of words to keep
max_sequence_length = 300 # Maximum length of sequences

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_text_train)

# Convert text to sequences
X_text_train_seq = tokenizer.texts_to_sequences(X_text_train)
X_text_test_seq = tokenizer.texts_to_sequences(X_text_test)

# Pad sequences
X_text_train_padded = pad_sequences(X_text_train_seq, maxlen=max_sequence_length, padding='post')
X_text_test_padded = pad_sequences(X_text_test_seq, maxlen=max_sequence_length, padding='post')

# Get vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")

# Build the model with both text and numerical features
embedding_dim = 128

# Text input branch
text_input = tf.keras.Input(shape=(max_sequence_length,), name='text_input')
embedding = Embedding(vocab_size, embedding_dim, input_length=max_sequence_length)(text_input)
lstm_1 = Bidirectional(LSTM(64, return_sequences=True))(embedding)
dropout_1 = Dropout(0.3)(lstm_1)
lstm_2 = Bidirectional(LSTM(32))(dropout_1)
dropout_2 = Dropout(0.3)(lstm_2)
text_features = Dense(64, activation='relu')(dropout_2)

# Numerical input branch
numerical_input = tf.keras.Input(shape=(len(numerical_features),), name='numerical_input')
numerical_dense_1 = Dense(32, activation='relu')(numerical_input)
numerical_dense_2 = Dense(16, activation='relu')(numerical_dense_1)
numerical_features = Dense(8, activation='relu')(numerical_dense_2)

# Combine text and numerical features
combined = tf.keras.layers.concatenate([text_features, numerical_features])
dense_1 = Dense(64, activation='relu')(combined)
dropout_3 = Dropout(0.3)(dense_1)
dense_2 = Dense(32, activation='relu')(dropout_3)
output = Dense(1)(dense_2) # Output layer for regression

# Create the model
model = tf.keras.Model(inputs=[text_input, numerical_input], outputs=output)

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Print model summary
model.summary()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

# Train the model
print("Training the model")
history = model.fit(
    [X_text_train_padded, X_numerical_train_scaled], y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')

# Evaluate the model
print("Evaluating the model")
loss, mae = model.evaluate([X_text_test_padded, X_numerical_test_scaled], y_test_scaled)
print(f"Test Loss: {loss}")
print(f"Test MAE: {mae}")

def save_model_components(model, tokenizer, numerical_scaler, y_scaler, max_sequence_length, numerical_features):
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model.save('models/movie_revenue_model.h5')
    
    # Save the tokenizer
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save the numerical scaler
    with open('models/numerical_scaler.pickle', 'wb') as handle:
        pickle.dump(numerical_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save the y scaler
    with open('models/y_scaler.pickle', 'wb') as handle:
        pickle.dump(y_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save the max sequence length and numerical features
    with open('models/config.pickle', 'wb') as handle:
        config = {
            'max_sequence_length': max_sequence_length,
            'numerical_features': numerical_features
        }
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Model and components saved successfully")

# Make predictions on test data
y_pred_scaled = model.predict([X_text_test_padded, X_numerical_test_scaled])
y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# Show 3 test predictions
print("\n=== Test Predictions ===")
for i in range(3):
    idx = np.random.randint(0, len(X_text_test))
    overview = df.loc[df['processed_overview'] == X_text_test[idx], 'overview'].values[0]
    actual_revenue = y_test[idx]
    predicted_revenue = y_pred[idx]
    
    # Skip displaying numerical features to avoid KerasTensor issue
    print(f"\nMovie Overview: {overview[:200]}")
    print(f"Actual Revenue: ${actual_revenue:,.2f}")
    print(f"Predicted Revenue: ${predicted_revenue:,.2f}")
    print(f"Difference: ${abs(actual_revenue - predicted_revenue):,.2f}")

# Function to predict revenue for user input
def predict_user_input(overview, contrast, avg_blue, avg_green, avg_red, texture, edge_detection, anomaly_score, people):
    # Preprocess the text input
    processed_input = preprocess_text(overview)
    
    # Convert to sequence
    input_seq = tokenizer.texts_to_sequences([processed_input])
    
    # Pad sequence
    input_padded = pad_sequences(input_seq, maxlen=max_sequence_length, padding='post')
    
    # Prepare numerical features
    numerical_input = np.array([[contrast, avg_blue, avg_green, avg_red, texture, edge_detection, anomaly_score, people]])
    numerical_input_scaled = numerical_scaler.transform(numerical_input)
    
    # Make prediction
    pred_scaled = model.predict([input_padded, numerical_input_scaled])
    prediction = y_scaler.inverse_transform(pred_scaled).flatten()[0]
    
    return prediction

'''
# Input a movie overview and numerical features for prediction
print("\n=== User Input Prediction ===")
print("Enter a movie overview to predict its revenue:")
user_overview = input()
print("Enter contrast value:")
user_contrast = float(input())
print("Enter average blue value:")
user_avg_blue = float(input())
print("Enter average green value:")
user_avg_green = float(input())
print("Enter average red value:")
user_avg_red = float(input())
print("Enter texture value:")
user_texture = float(input())
print("Enter edge detection value:")
user_edge_detection = float(input())
print("Enter anomaly score value:")
user_anomaly_score = float(input())
print("Enter people count:")
user_people = float(input())

predicted_revenue = predict_user_input(
    user_overview, 
    user_contrast, 
    user_avg_blue, 
    user_avg_green, 
    user_avg_red, 
    user_texture,
    user_edge_detection,
    user_anomaly_score,
    user_people
)
print(f"Predicted Revenue: ${predicted_revenue:,.2f}")
'''

# Save the model and components for future use
save_model_components(model, tokenizer, numerical_scaler, y_scaler, max_sequence_length, numerical_features)
