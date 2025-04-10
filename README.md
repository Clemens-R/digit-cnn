# Handwritten Digit Recognition with CNN

A project for recognizing handwritten digits using Convolutional Neural Networks (CNN) and the MNIST dataset.

## Project Overview

This project implements a handwritten digit recognition system using TensorFlow and Keras. It includes:

- A CNN model trained on the MNIST dataset
- Data augmentation to improve model robustness
- A testing script to evaluate model performance
- An interactive GUI application for drawing and recognizing digits in real-time

## Files

- `cnn_mnist_training.py`: Trains the CNN model on the MNIST dataset with data augmentation
- `testing_cnn.py`: Tests the trained model on the MNIST test set with visualizations
- `handwriting.py`: Interactive GUI application for drawing and recognizing digits

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install tensorflow numpy matplotlib pillow
```

## Usage

### Training the Model

Run the training script to train the CNN model:
```bash
python cnn_mnist_training.py
```

The trained model will be saved as `augmented_mnist_cnn_model.h5`.

### Testing the Model

Run the testing script to evaluate the model on the MNIST test set:
```bash
python testing_cnn.py
```

### Interactive Digit Recognition

Launch the interactive drawing application:
```bash
python handwriting.py
```

- Draw a digit on the canvas using your mouse
- The application will automatically recognize the digit in real-time
- Click "clear all" to erase your drawing and start over

## Model Architecture

The CNN model architecture consists of:
- 3 convolutional layers with ReLU activation
- 2 max pooling layers
- Dense layers for classification
- Softmax output layer for 10 classes (digits 0-9)

## License

MIT License
Feel free to use, adapt and share
