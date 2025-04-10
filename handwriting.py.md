# Handwriting Recognition Application

This script provides an interactive GUI application for drawing and recognizing handwritten digits in real-time.

## Overview

The application performs the following functions:
1. Creates a Tkinter GUI window with a drawing canvas
2. Loads the pre-trained CNN model
3. Allows users to draw digits using mouse movements
4. Processes the drawn image to match the MNIST format
5. Makes real-time predictions using the CNN model
6. Displays the recognized digit and confidence scores

## GUI Components

The application interface consists of:
- A 280×280 pixel drawing canvas
- A "clear all" button to reset the canvas
- Two visualization panels:
  - Left panel: Processed 28×28 image of the drawn digit with prediction
  - Right panel: Bar chart showing confidence scores for all possible digits (0-9)

## Drawing Functionality

- Users can draw on the canvas using mouse movements
- Drawing is performed with a black brush on a white background
- The application automatically triggers predictions as the user draws

## Prediction Process

The prediction process involves:
- Resizing the drawn image to 28×28 pixels to match the MNIST format
- Normalizing pixel values to the 0-1 range
- Inverting colors (MNIST has white digits on black background)
- Running the image through the pre-trained CNN model
- Updating the visualization panels with the prediction results

## Threading

The application uses a separate thread for prediction to keep the GUI responsive during model inference.

## Model Loading

The script loads the model from "augmented_mnist_cnn_model.h5", which should be created by the training script.