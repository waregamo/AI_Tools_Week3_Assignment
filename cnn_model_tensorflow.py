import tensorflow as tf
import numpy as np
import os

# 1. Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., tf.newaxis] / 255.0
x_test = x_test[..., tf.newaxis] / 255.0

# 2. Define and train model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Create models directory 
os.makedirs("models", exist_ok=True)

# 4. Add ModelCheckpoint callback to save during training
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/mnist_cnn.keras",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# 5. Train model
print("\nTraining model...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[checkpoint]
)

# 6. Save final model (in both formats)
model.save("models/mnist_cnn.keras")
model.save("models/mnist_cnn.h5")
print("\nModels saved successfully:")
print(f"- models/mnist_cnn.keras")
print(f"- models/mnist_cnn.h5")

# 7. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc:.2%}")
