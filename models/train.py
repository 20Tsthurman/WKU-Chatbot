from .chatbot_model import create_model
import pickle
from data.preprocessing import load_and_preprocess_data

def train_model(data_file, input_length):
    # Load and preprocess the data
    features, labels, tokenizer = load_and_preprocess_data(data_file)
    
    # Create the model
    model = create_model(input_length, num_classes=len(set(labels)))
    
    # Train the model
    model.fit(features, labels, epochs=20)

    # Save the model and tokenizer
    model.save('chatbot_model.h5')
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
