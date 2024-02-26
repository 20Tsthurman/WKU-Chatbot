from .chatbot_model import create_model
import pickle
import numpy as np  # Ensure numpy is imported
from data.preprocessing import load_and_preprocess_data

def train_model(data_file, input_length):
    # Load and preprocess the data
    features, labels, tokenizer = load_and_preprocess_data(data_file)
    
    # Determine the number of unique classes
    num_classes = np.unique(labels).size
    
    # Create the model with the correct number of input length and classes
    model = create_model(input_length, num_classes=num_classes)

    # Train the model
    model.fit(features, labels, epochs=20)

    # Save the model using the SavedModel format
    model.save('chatbot_model')  # Updated to use SavedModel format

    # Save the tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Note: Consider evaluating your model on a test set to assess its generalization capabilities
