import tensorflow as tf
import pickle
from keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = tf.keras.models.load_model('chatbot_model')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess_input(user_query, max_length=4):  # Adjust max_length to match your model's expected input length
    # Tokenize and pad the user query
    sequence = tokenizer.texts_to_sequences([user_query])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    return padded_sequence

def get_response(user_query):
    # Preprocess the user query
    processed_query = preprocess_input(user_query, max_length=4)  # Ensure this matches the expected input length of your model
    
    # Make a prediction
    prediction = model.predict(processed_query)
    
    # Decode the prediction into a human-readable response
    predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
    
    # Enhanced responses dictionary
    responses = {
        0: "Hello! How can I assist you today?",
        1: "I'm a chatbot designed to answer questions about our services. How can I help you?",
        2: "I'm not sure how to respond to that. Can you try asking in a different way?",
        # Add more responses for each class your model can predict
    }
    
    response = responses.get(predicted_class_index, "I'm sorry, I don't quite understand. Could you rephrase your question?")
    
    return response

# Example usage
if __name__ == "__main__":
    print("Chatbot is ready to talk! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        print("Bot:", get_response(user_input))
