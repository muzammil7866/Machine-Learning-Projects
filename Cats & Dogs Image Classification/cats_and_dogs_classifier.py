"""
Cats and Dogs Classification using Convolutional Neural Network (CNN)
This script builds and trains a CNN model to classify images of cats and dogs.
"""

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
import matplotlib.pyplot as plt
import cv2
import os

print("="*70)
print("CATS AND DOGS CLASSIFICATION USING CNN")
print("="*70)

# Data generators - Loading and preprocessing images
print("\nLoading training dataset...")
train_ds = keras.utils.image_dataset_from_directory(
    directory='archive/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
    seed=42
)
print("✓ Training dataset loaded")

print("Loading validation dataset...")
validation_ds = keras.utils.image_dataset_from_directory(
    directory='archive/test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
    seed=42
)
print("✓ Validation dataset loaded")

# Normalize the image data
print("\nNormalizing image data...")

def process(image, label):
    """Normalize images to [0, 1] range."""
    image = tf.cast(image / 255., tf.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)
print("✓ Data normalization complete")

print("\n" + "="*70)
print("BUILDING CNN MODEL")
print("="*70 + "\n")

# Create CNN model
model = Sequential([
    # Block 1
    Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', 
           input_shape=(256, 256, 3), name='conv1'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='pool1'),
    
    # Block 2
    Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu', name='conv2'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='pool2'),
    
    # Block 3
    Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu', name='conv3'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='pool3'),
    
    # Flattening and Dense layers
    Flatten(),
    Dense(128, activation='relu', name='dense1'),
    Dropout(0.1),
    Dense(64, activation='relu', name='dense2'),
    Dropout(0.1),
    Dense(1, activation='sigmoid', name='output')  # Binary classification
])

# Display model architecture
print("Model Architecture:")
model.summary()

# Compile the model
print("\nCompiling model...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("✓ Model compiled successfully")

# Train the model
print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70 + "\n")

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=validation_ds,
    verbose=1
)

print("\n✓ Training completed!")

# Evaluate the model
print("\n" + "="*70)
print("EVALUATING MODEL")
print("="*70 + "\n")

val_loss, val_accuracy = model.evaluate(validation_ds, verbose=0)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save the model
print("\nSaving model...")
model.save('cats_dogs_cnn_model.h5')
print("✓ Model saved as 'cats_dogs_cnn_model.h5'")

# Plotting training history
print("\n" + "="*70)
print("PLOTTING TRAINING HISTORY")
print("="*70 + "\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy
axes[0].plot(history.history['accuracy'], color='blue', label='Train Accuracy', linewidth=2, marker='o')
axes[0].plot(history.history['val_accuracy'], color='red', label='Validation Accuracy', linewidth=2, marker='s')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Accuracy', fontsize=11)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Loss
axes[1].plot(history.history['loss'], color='blue', label='Train Loss', linewidth=2, marker='o')
axes[1].plot(history.history['val_loss'], color='red', label='Validation Loss', linewidth=2, marker='s')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Loss', fontsize=11)
axes[1].set_title('Model Loss Over Epochs', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_cnn.png', dpi=100, bbox_inches='tight')
print("Saved: training_history_cnn.png")
plt.show()

# Make predictions on sample images
print("\n" + "="*70)
print("MAKING PREDICTIONS ON SAMPLE IMAGES")
print("="*70 + "\n")

# Function to predict on a single image
def predict_image(image_path, model, img_size=256):
    """
    Predict whether an image is a cat or dog.
    """
    try:
        # Read and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0  # Normalize
        img = tf.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = model.predict(img, verbose=0)[0][0]
        
        # Interpret prediction
        if prediction > 0.5:
            return "Dog", prediction
        else:
            return "Cat", 1 - prediction
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Test with available sample images
print("Testing predictions on sample images from training data:")
print("-" * 70)

sample_dirs = [
    ('archive/train/cats', 'cats'),
    ('archive/train/dogs', 'dogs'),
]

predictions_made = 0
for dir_path, label in sample_dirs:
    if os.path.exists(dir_path):
        files = os.listdir(dir_path)[:2]  # Take first 2 images from each category
        for file in files:
            image_path = os.path.join(dir_path, file)
            result = predict_image(image_path, model)
            if result:
                predicted_class, confidence = result
                print(f"Image: {file:25} | Actual: {label:5} | Predicted: {predicted_class:5} | Confidence: {confidence:.4f}")
                predictions_made += 1

if predictions_made == 0:
    print("Note: No sample images found for testing. Ensure archive/train directory exists with cat and dog images.")

print("\n" + "="*70)
print("CATS AND DOGS CLASSIFICATION COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nModel Information:")
print(f"  - Total epochs trained: {len(history.history['accuracy'])}")
print(f"  - Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  - Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"  - Total parameters: {model.count_params():,}")

# Notes on reducing overfitting
print("\n" + "="*70)
print("TECHNIQUES USED TO REDUCE OVERFITTING:")
print("="*70)
print("1. ✓ Added more data (using image data from directory)")
print("2. ✓ Batch Normalization")
print("3. ✓ Dropout layers (0.1 rate)")
print("4. ✓ Model regularization through dropout")
print("\nOther techniques that can be applied:")
print("  - Data augmentation (rotation, flip, zoom, etc.)")
print("  - L1/L2 regularization")
print("  - Early stopping with validation monitoring")
print("  - Reducing model complexity (fewer layers)")
