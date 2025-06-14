import tensorflow as tf

# Load the old model with compatibility fix
custom_objects = {'InputLayer': tf.keras.layers.InputLayer}
model = tf.keras.models.load_model(
    "mnist_cnn.h5",
    custom_objects=custom_objects,
    compile=False
)
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.save("mnist_cnn.keras")  # Save in new format