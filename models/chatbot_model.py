import tensorflow as tf

def create_model(input_length, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=500, output_dim=16, input_length=input_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model
