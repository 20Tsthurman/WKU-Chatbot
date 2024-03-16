import tensorflow as tf

def create_model(input_length, num_classes):
    model = tf.keras.Sequential([
        # Embedding layer to handle text input, transforming word indices into dense vectors
        tf.keras.layers.Embedding(input_dim=1000,  # Size of the vocabulary
                                  output_dim=16,  # Dimension of the dense embedding
                                  input_length=input_length),  # Length of input sequences
        
        # Pooling layer to reduce the dimensionality and to handle variable input sequence lengths
        tf.keras.layers.GlobalAveragePooling1D(),
        
        # Dense layer for learning non-linear relationships
        tf.keras.layers.Dense(24, activation='relu'),
        
        # Output layer with a softmax activation to output probabilities for each class
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model with a loss function suited for multi-class classification, an optimizer, and a metric to monitor
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model
