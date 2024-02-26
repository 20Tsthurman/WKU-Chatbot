import tensorflow as tf

def create_model(input_length, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(1000, 16, input_length=4),  # Adjusted input_length to 4
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
