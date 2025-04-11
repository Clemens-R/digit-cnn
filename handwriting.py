import tkinter as tk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time 
import threading

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw

# Load the pre-trained CNN model
# This model was trained on the MNIST dataset with data augmentation
model = tf.keras.models.load_model("mnist_cnn_model.h5") 

# Create the main application window
window = tk.Tk()
window.title("handwriting")

# Create a frame for organizing the canvas
frame = tk.Frame(window)
frame.pack(pady=10)

# Create a canvas for drawing digits
# Size 280x280 pixels (will be resized to 28x28 for the model)
canvas = tk.Canvas(frame, width=280, height=280, bg="white", relief="solid", borderwidth=2)
canvas.pack()

# Create a PIL Image object to store the drawn digit
# This will be processed and fed to the model
image = Image.new("L", (280, 280), "white")  # L mode = 8-bit grayscale
draw = ImageDraw.Draw(image)

# Flag to control prediction thread
predicion_running = False

# Function to handle mouse drawing events
def draw_digit(event):
    x = event.x
    y = event.y
    r = 10  # Brush radius
    
    # Draw circle on both the visible canvas and the PIL image
    canvas.create_oval(x-r, y-r, x+r, y+r, fill = "black", outline = "black")
    draw.ellipse([x-r, y-r, x+r, y+r], fill = "black")

    # Start the prediction thread if not already running
    global predicion_running
    if not predicion_running:
        predicion_running = True
        threading.Thread(target = predict, daemon = True).start()

# Bind the drawing function to mouse movement while left button is pressed
canvas.bind("<B1-Motion>", draw_digit)

# Function that runs in a separate thread to make predictions
def predict():
    global predicion_running
    global last_prediction

    while True:
        time.sleep(0.1)  # Short delay to avoid excessive predictions

        # Process the image for the model:
        # 1. Resize to 28x28 (same as MNIST)
        img_resized = image.resize((28, 28))
        
        # 2. Convert to numpy array and normalize pixel values
        img_array = np.array(img_resized) / 255.0
        
        # 3. Invert colors (MNIST has white digits on black background)
        img_array = 1 - img_array
        
        # 4. Reshape for the model (add batch and channel dimensions)
        img_array = img_array.reshape(1, 28, 28, 1)

        # Make prediction using the model
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)  # Get digit with highest probability

        # Schedule GUI update on the main thread
        window.after(0, lambda: update_gui(img_resized, prediction, predicted_digit))

        # End the prediction thread
        predicion_running = False
        break

# Function to update the GUI with prediction results
def update_gui(img_resized, prediction, predicted_digit):

    # Update the image display
    ax_img.clear()
    ax_img.imshow(img_resized, cmap="gray")
    ax_img.set_title(f"Zahl: {predicted_digit}")
    ax_img.axis("off")

    # Update the probability bar chart
    ax_bar.clear()
    ax_bar.bar(range(10), prediction[0], color="blue")  # Show probability for each digit
    ax_bar.set_xticks(range(10))  # Set x-axis ticks for digits 0-9
    ax_bar.set_ylim([0, 1])  # Set y-axis range for probabilities (0-1)
    ax_bar.set_title("Prediction in %")

    # Refresh the matplotlib canvas
    canvas_plot.draw()

# Function to clear the drawing canvas and reset displays
def clear_canvas():
    global predicion_running

    # Clear the visible canvas
    canvas.delete("all")
    
    # Clear the PIL image
    draw.rectangle([0, 0, 280, 280], fill="white")
    
    # Clear the prediction displays
    ax_img.clear()
    ax_bar.clear()
    ax_img.set_title("")
    ax_bar.set_title("")
    canvas_plot.draw()
    
    # Reset the prediction flag
    predicion_running = False

# Create a frame for the buttons
btn_frame = tk.Frame(window)
btn_frame.pack(pady=10)

# Add a clear button
btn_clear = tk.Button(btn_frame, text="clear all", command=clear_canvas)
btn_clear.pack(side="right", padx=10)

# Create matplotlib figure with two subplots side by side
# Left: processed image, Right: probability bars
fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(8, 4))

# Embed matplotlib figure in the tkinter window
canvas_plot = FigureCanvasTkAgg(fig, master = window)
canvas_plot.get_tk_widget().pack()

# Start the main application loop
window.mainloop()
