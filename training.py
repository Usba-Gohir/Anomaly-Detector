import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 🔹 Enable GPU if available
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("🚀 Using GPU for training!")
else:
    print("⚠️ No GPU detected. Using CPU instead.")

# 🔹 Constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

# 🔹 Preprocess Images (Extract Apples from Background)
def extract_apple(image):
    """ Extracts the apple from the background using HSV thresholding """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([10, 40, 40])  # Adjusted HSV range for apples
    upper_bound = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)  # Create mask for apple
    mask = cv2.medianBlur(mask, 5)  # Reduce noise
    
    # Find contours and extract the largest one (the apple)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return image[y:y+h, x:x+w]  # Crop the apple region
    
    return image  # If no contour is found, return original image

# 🔹 Load and Process Images
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)  # Read image
            if img is None:
                continue  # Skip corrupted images
            
            img = extract_apple(img)  # Extract apple from background
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, IMAGE_SIZE)  # Resize
            img = img.astype("float32") / 255.0  # Normalize
            images.append(img)
    
    images = np.array(images, dtype=np.float32)  # Convert list to NumPy array
    if images.shape[0] == 0:
        raise ValueError(f"Error: No valid images found in {directory}.")
    
    return images

# 🔹 Load fresh images only (NO rotten apples in training)
train_images = load_images("C:/Users/usbag/Desktop/Anomaly Detector/freshapples")

# 🔹 Data Augmentation (Prevents Overfitting)
datagen = ImageDataGenerator(
    rotation_range=30,  # More aggressive rotations
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,  # More zoom
    horizontal_flip=True,
    fill_mode="nearest"
)

# 🔹 Verify Dataset Shape
if train_images.shape[1:] != (128, 128, 3):
    raise ValueError(f"Dataset shape mismatch. Expected (batch, 128, 128, 3), but got {train_images.shape}")

# 🔹 Define Autoencoder Model
input_img = Input(shape=(128, 128, 3))

# Encoder
x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)

# Bottleneck
x = Flatten()(x)
x = Dense(256, activation="relu")(x)  # Bottleneck layer
x = Dense(16 * 16 * 128, activation="relu")(x)
x = Reshape((16, 16, 128))(x)

# Decoder
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)  # Output layer

autoencoder = Model(input_img, x)

# 🔥 **Fix the Loss Function to Mean Squared Error**
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

# 🔹 Train Autoencoder & Save Loss History
history = autoencoder.fit(
    datagen.flow(train_images, train_images, batch_size=BATCH_SIZE),  
    epochs=EPOCHS,
    steps_per_epoch=max(len(train_images) // BATCH_SIZE, 1),  # Avoid division errors
    shuffle=True
)

# 🔹 Save Model
autoencoder.save("anomaly_detector_model.keras")

# 🔹 Save Training Loss
loss_df = pd.DataFrame({"Epoch": range(1, EPOCHS + 1), "Loss": history.history["loss"]})
loss_df.to_csv("training_loss.csv", index=False)

# 🔹 Plot Loss Curve
plt.plot(loss_df["Epoch"], loss_df["Loss"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid()
plt.show()

print("🎉 Training Completed & Model Saved Successfully!")