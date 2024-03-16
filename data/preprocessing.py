import json
import re
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Ensure NLTK is installed for stemming
import nltk
from nltk.stem import PorterStemmer

# Download the punkt tokenizer for word tokenization
nltk.download('punkt')

def clean_text(text):
    """Basic text cleaning"""
    text = text.lower()  # Lowercase text
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove punctuation
    return text

def stem_text(text):
    """Stem text to its root form"""
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

def load_and_preprocess_data(filename):
    # Load JSON data
    with open(filename) as file:
        data = json.load(file)

    patterns = []
    tags = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            cleaned_pattern = clean_text(pattern)  # Clean text
            stemmed_pattern = stem_text(cleaned_pattern)  # Stem text
            patterns.append(stemmed_pattern)  # Add cleaned and stemmed pattern to list
            tags.append(intent['tag'])

    # Initialize and fit the tokenizer on the patterns
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=2000, oov_token="<OOV>")
    tokenizer.fit_on_texts(patterns)
    sequences = tokenizer.texts_to_sequences(patterns)

    # Pad sequences to ensure uniform length
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

    label_encoder = LabelEncoder()
    integer_encoded_tags = label_encoder.fit_transform(tags)

    max_length = max([len(seq) for seq in sequences])  # Use max length for dynamic padding
    return padded_sequences, integer_encoded_tags, tokenizer, max_length
