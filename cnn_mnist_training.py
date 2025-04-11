import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
# MNIST contains 70,000 grayscale images of handwritten digits (28x28 pixels)
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values from [0,255] to [0,1] range for better training performance
x_train = x_train / 255.0
x_test = x_test / 255.0

# Print dataset information
print("Anzahl Trainingsdaten:", len(x_train))  # 60,000 training samples
print("Anzahl Testdaten", len(x_test))       # 10,000 test samples

# Create data augmentation generator to artificially increase the training set size
# This helps improve model generalization and prevents overfitting
datagen = ImageDataGenerator(
    rotation_range = 15,         # Randomly rotate images by up to 15 degrees
    width_shift_range = 0.1,     # Randomly shift images horizontally by up to 10%
    height_shift_range = 0.1,    # Randomly shift images vertically by up to 10%
    shear_range = 0.1,           # Randomly apply shearing transformations
    zoom_range = 0.1,            # Randomly zoom images by up to 10%
)

# Create the CNN model architecture
model = keras.Sequential([
    # First convolutional layer: 32 filters of size 3x3 with ReLU activation
    # Input shape: 28x28 pixel images with 1 color channel (grayscale)
    keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape =(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),  # Max pooling layer: Reduces spatial dimensions by half
    
    # Second convolutional layer: 64 filters of size 3x3 with ReLU activation
    keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D((2, 2)),  # Another max pooling layer
    
    # Third convolutional layer: 64 filters of size 3x3 with ReLU activation
    keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    
    # Flatten layer: Converts the 2D feature maps to 1D feature vectors
    keras.layers.Flatten(),
    
    # First dense (fully connected) layer: 64 neurons with ReLU activation
    keras.layers.Dense(64, activation = 'relu'),
    
    # Output layer: 10 neurons (one for each digit 0-9) with softmax activation
    # Softmax ensures all output values sum to 1 (probability distribution)
    keras.layers.Dense(10, activation = 'softmax')
])

# Add a dimension to the input data for the color channel (required by Conv2D layers)
x_train = np.expand_dims(x_train, axis=-1)  # Shape becomes (60000, 28, 28, 1)

# Fit the data generator to the training data
datagen.fit(x_train)

# Compile the model
model.compile(optimizer = 'adam',                      # Adam optimizer for efficient gradient descent
              loss = 'sparse_categorical_crossentropy', # Appropriate loss function for multi-class classification
              metrics = ['accuracy'])                   # Track accuracy during training

# Train the model using the data generator
model.fit(datagen.flow(x_train, y_train, batch_size = 8), # Generate batches of augmented data
          epochs = 10)                                    # Run for 10 complete passes through the dataset

# Evaluate model on test data
# This shows how well the model can generalize to unseen data
model.evaluate(x_test, y_test)

# Save the trained model to a file for later use
model.save("mnist_cnn_model.h5")
print("Model is trained and saved")
