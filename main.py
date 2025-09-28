# ========================================
# CNN on MNIST + Serving with FastAPI
# ========================================


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

# =========================
# 1. Train CNN on MNIST
# =========================

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize to [0,1] range
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension (28,28) -> (28,28,1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")  # 10 classes for digits 0–9
])

# Compile model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train model
print("Training CNN on MNIST...")
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Save model
model.save("mnist_cnn")
print("Model saved as mnist_cnn")

# =========================
# 2. FastAPI App
# =========================

# Load trained model for inference
model = keras.models.load_model("mnist_cnn")

# Initialize FastAPI
app = FastAPI(title="MNIST CNN API", description="Digit recognition using CNN on MNIST", version="1.0")

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    """
    Predict digit from an uploaded image.
    Expected: grayscale 28x28 image (or any image, resized internally).
    """

    # Read uploaded file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28

    # Preprocess for model
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1,28,28,1)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    return {"predicted_digit": predicted_class, "confidence": confidence}

# To run FastAPI server:
# uvicorn filename:app --reload
