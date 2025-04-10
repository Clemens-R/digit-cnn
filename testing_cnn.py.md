# Testing CNN

This script tests the trained CNN model on the MNIST test dataset with visualizations.

## Overview

The script performs the following operations:
1. Loads the pre-trained CNN model
2. Loads and normalizes the MNIST test dataset
3. Creates an interactive visualization for model predictions
4. Displays 20 random test images with their predictions and confidence scores

## Visualization Features

The visualization consists of:
- A two-panel figure arrangement
- The top panel shows the test digit image with prediction results
- The bottom panel shows a bar chart of prediction probabilities for all 10 digits
- Each image is displayed for 5 seconds (1 second for the image, 4 seconds for the probability display)

## Model Loading

The script loads the model from "mnist_cnn_model.h5", which should be created by the training script.

## Interactive Display

The script uses matplotlib's interactive mode to update the display in real-time as it cycles through the test images.

## Output Analysis

For each test image, the script shows:
- The original grayscale image
- The model's predicted digit
- The actual (true) digit label
- A bar chart showing confidence scores for all possible digits (0-9)