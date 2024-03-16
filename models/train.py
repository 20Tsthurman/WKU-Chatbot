from .chatbot_model import create_model
import pickle
import numpy as np
from data.preprocessing import load_and_preprocess_data
import tensorflow as tf  # Import TensorFlow directly
from sklearn.utils.class_weight import compute_class_weight

def train_model(data_file):
    print("Loading and preprocessing data...")
    # Load and preprocess the data
    features, labels, tokenizer, max_length = load_and_preprocess_data(data_file)
    print("Data loaded and preprocessed.")
    
    # Determine the number of unique classes
    num_classes = np.unique(labels).size
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Create the model with the correct number of input length and classes
    model = create_model(max_length, num_classes=num_classes)  # Use dynamic max_length
    
    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    
    # Define learning rate scheduler
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.96,
        staircase=True)
    
    # Compile the model with the learning rate scheduler
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model with a validation split
    model.fit(features, labels, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stopping, model_checkpoint], shuffle=True, verbose=1, class_weight=class_weight_dict)

    # Load the best model saved by ModelCheckpoint
    model.load_weights('best_model.h5')

    # Save the final model
    model.save('chatbot_model')

    # Save the tokenizer and max_length
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('max_length.pickle', 'wb') as handle:
        pickle.dump(max_length, handle, protocol=pickle.HIGHEST_PROTOCOL)
