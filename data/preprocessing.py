import json
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# TensorFlow Keras imports for text processing
Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

def load_and_preprocess_data(filename):
    # Load JSON data
    with open(filename) as file:
        data = json.load(file)

    # Initialize lists to hold patterns and tags
    patterns = []
    tags = []

    # Loop through each intent in the JSON data
    for intent in data['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)  # Add pattern to patterns list
            tags.append(intent['tag'])  # Add corresponding tag to tags list

    # Initialize and fit the tokenizer on the patterns
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
    tokenizer.fit_on_texts(patterns)
    sequences = tokenizer.texts_to_sequences(patterns)

    # Pad the sequences to ensure uniform length
    padded_sequences = pad_sequences(sequences, padding='post')

    # Use LabelEncoder to convert tags to integers
    label_encoder = LabelEncoder()
    integer_encoded_tags = label_encoder.fit_transform(tags)

    max_length = padded_sequences.shape[1]  # Get the length of the longest sequence
    return padded_sequences, integer_encoded_tags, tokenizer, max_length