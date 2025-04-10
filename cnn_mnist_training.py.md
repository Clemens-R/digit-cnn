# CNN MNIST Training

This script trains a Convolutional Neural Network (CNN) on the MNIST handwritten digits dataset with data augmentation.

## Overview

The script performs the following operations:
1. Loads and normalizes the MNIST dataset
2. Creates a data augmentation pipeline for training images
3. Builds a CNN model architecture 
4. Trains the model with the augmented data
5. Evaluates the model on the test set
6. Saves the trained model to disk

## Data Augmentation

Data augmentation is used to artificially expand the training dataset by applying various transformations:
- Random rotations (±15 degrees)
- Width and height shifts (±10%)
- Shear transformations (±10%)
- Zoom range variations (±10%)

## Model Architecture

The CNN model architecture consists of:
- Input layer for 28×28 grayscale images
- Three convolutional layers with ReLU activation
- Two max pooling layers
- Flatten layer
- Two dense layers, including softmax output layer for 10 classes

## Training Parameters

- Optimizer: Adam
- Loss function: Sparse categorical crossentropy
- Batch size: 8
- Training epochs: 10

## Output

The trained model is saved as "augmented_mnist_cnn_model.h5" for later use in the testing and handwriting applications.