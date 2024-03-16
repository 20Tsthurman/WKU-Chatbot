from .chatbot_model import create_model
import pickle
import numpy as np
from data.preprocessing import load_and_preprocess_data
import tensorflow as tf  # Import TensorFlow directly

def train_model(data_file):
    print("Loading and preprocessing data...")
    # Load and preprocess the data
    features, labels, tokenizer, max_length = load_and_preprocess_data(data_file)
    print("Data loaded and preprocessed.")
    # Load and preprocess the data
    features, labels, tokenizer, max_length = load_and_preprocess_data(data_file)
    
    # Determine the number of unique classes
    num_classes = np.unique(labels).size
    
    # Create the model with the correct number of input length and classes
    model = create_model(max_length, num_classes=num_classes)  # Use dynamic max_length

    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

    # Train the model with a validation split
    model.fit(features, labels, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint], shuffle=True, verbose=1)

    # Load the best model saved by ModelCheckpoint
    model.load_weights('best_model.h5')

    # Save the final model
    model.save('chatbot_model')

    # Save the tokenizer
# Save max_length
    with open('max_length.pickle', 'wb') as handle:
        pickle.dump(max_length, handle, protocol=pickle.HIGHEST_PROTOCOL)

