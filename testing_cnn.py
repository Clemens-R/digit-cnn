import numpy as np
import time 
import matplotlib
matplotlib.use("TkAgg")  # Set the backend for matplotlib to TkAgg for GUI support
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Load the previously trained CNN model from file
model = load_model("mnist_cnn_model.h5")

# Load the MNIST test dataset
# We only need the test data (x_test, y_test) here, so we use underscores for the training data
(_, _), (x_test, y_test) = mnist.load_data()

# Normalize the image pixel values from [0,255] to [0,1] range
# This is crucial as the model was trained on normalized data
x_test = x_test / 255.0

# Enable interactive mode for matplotlib
# This allows the plot to be updated without blocking the script execution
plt.ion()

# Create a figure with two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))

# Loop through 20 test images
for i in range(20):

    # Get a test image and its true label
    image = x_test[i]
    true_label = y_test[i]

    # Make a prediction with our model
    # Reshape the image to match the model's input shape (1, 28, 28, 1)
    predictions = model.predict(image.reshape(1, 28, 28, 1))[0]
    predicted_label = np.argmax(predictions)  # Get the digit with highest probability

    # Clear previous plots
    ax1.clear()
    ax2.clear()
    
    # Display the current test image in the top subplot
    ax1.imshow(image, cmap="gray")
    ax1.axis("off")  # Hide the axes
    ax1.set_title(f"Prediction: {predicted_label}, true number: {true_label}", fontsize = 18)
    plt.draw()
    plt.pause(1)  # Pause for 1 second to show the image

    # Display the prediction probabilities as a bar chart in the bottom subplot
    ax2.bar(range(10), predictions, color = 'blue', alpha = 0.6)
    ax2.set_xticks(range(10))  # Set x-axis ticks for digits 0-9
    ax2.set_ylim(0, 1)  # Set y-axis range for probabilities (0-1)
    ax2.set_ylabel("Properbility")

    plt.draw()
    plt.pause(4)  # Pause for 4 seconds to show the probability bars

# Disable interactive mode when finished
plt.ioff()
plt.show()
