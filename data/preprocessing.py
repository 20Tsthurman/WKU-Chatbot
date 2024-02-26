import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

def load_and_preprocess_data(filename):
    # Load data
    df = pd.read_csv(filename)
    
    # Assume the CSV has two columns: 'input' for the input and 'output' for the label
    sentences = df['input'].tolist()
    labels = df['output'].tolist()

    # Tokenize sentences
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, padding='post')

    # Convert labels to integers using LabelEncoder
    label_encoder = LabelEncoder()
    integer_encoded_labels = label_encoder.fit_transform(labels)

    # Return the padded sequences and integer encoded labels
    return padded_sequences, integer_encoded_labels, tokenizer
