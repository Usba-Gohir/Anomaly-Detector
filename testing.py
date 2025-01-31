import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load trained model
autoencoder = tf.keras.models.load_model("anomaly_detector_model.keras")

# Function to calculate reconstruction error
def calculate_reconstruction_error(original, reconstructed):
    original = original.numpy().squeeze()
    reconstructed = np.squeeze(reconstructed)
    mse_error = np.mean(np.square(original - reconstructed))
    return mse_error

# Load test data (fresh + rotten)
test_data = "C:/Users/usbag/Desktop/Anomaly Detector/test"

# Compute reconstruction error & generate metrics
errors = []
for image_path in os.listdir(test_data):
    image = preprocess_test_image(os.path.join(test_data, image_path))
    reconstructed = autoencoder.predict(image)
    error = calculate_reconstruction_error(image, reconstructed)
    errors.append(error)

# Determine adaptive threshold
threshold = np.percentile(errors, 95)

# Plot error distribution
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, alpha=0.7, label="Reconstruction Errors")
plt.axvline(threshold, color="red", linestyle="dashed", linewidth=2, label="Threshold")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()
