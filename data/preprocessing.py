import pandas as pd
import tensorflow as tf

Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences


def load_and_preprocess_data(filename):
    # Load data
    df = pd.read_csv(filename)
    
    # Assume the CSV has two columns: 'text' for the input and 'intent' for the label
    sentences = df['text'].tolist()
    labels = df['intent'].tolist()

    # Tokenize sentences
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, padding='post')

    return padded_sequences, labels, tokenizer
