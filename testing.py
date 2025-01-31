import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the trained model
model_path = "anomaly_detector_model.keras"
autoencoder = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# 🔹 **Preprocess a single test image (with slight augmentations)**
def preprocess_test_image(image_path, image_size=(128, 128)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = image / 255.0  # Normalize

    # 🔥 Apply minor augmentations to improve generalization
    image = tf.image.random_brightness(image, max_delta=0.1)  # Adjust brightness
    image = tf.image.random_contrast(image, 0.9, 1.1)  # Adjust contrast

    return tf.expand_dims(image, axis=0)  # Add batch dimension

# 🔹 **Calculate reconstruction error using MSE & SSIM**
def calculate_reconstruction_error(original, reconstructed):
    original = original.numpy().squeeze()  # Convert Tensor to NumPy
    reconstructed = np.squeeze(reconstructed)  # No need for `.numpy()`

    mse_error = np.mean(np.square(original - reconstructed))
    ssim_score = ssim(original, reconstructed, channel_axis=-1, data_range=1.0)

    return mse_error, 1 - ssim_score  # SSIM is similarity, so we use (1 - SSIM)

# 🔹 **Determine an adaptive threshold using percentiles**
def determine_threshold(errors, percentile=85):  # 🔥 Lowered threshold from 90% to 85%
    return np.percentile(errors, percentile)

# 🔹 **Test the autoencoder on a directory of images**
def test_model_on_directory(base_dir, image_size=(128, 128)):
    results = []
    all_errors = []

    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        print(f"\n🔍 Testing images in folder: {class_name}")

        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Preprocess the image
            test_image = preprocess_test_image(image_path, image_size)

            # Get the reconstruction from the model
            reconstructed_image = autoencoder.predict(test_image)
            mse_error, ssim_error = calculate_reconstruction_error(test_image, reconstructed_image)

            avg_error = (mse_error + ssim_error) / 2
            all_errors.append(avg_error)

            results.append({
                "Image": filename,
                "Folder": class_name,
                "Reconstruction Error": avg_error
            })

    return results, all_errors

# 🔹 **Save results to a CSV file**
def save_results_to_csv(results, output_file="test_results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"📂 Results saved to {output_file}")

# 🔹 **Calculate Accuracy, Precision, Recall, F1-Score**
def calculate_metrics(results, threshold):
    y_true = []
    y_pred = []

    for result in results:
        ground_truth = "Normal" if result["Folder"].lower() == "freshapples" else "Anomalous"

        # 🔥 **Dynamically adjust the threshold**
        adjusted_threshold = threshold * 0.95  # Reducing threshold slightly for better recall

        predicted_label = "Normal" if result["Reconstruction Error"] <= adjusted_threshold else "Anomalous"

        y_true.append(1 if ground_truth == "Anomalous" else 0)
        y_pred.append(1 if predicted_label == "Anomalous" else 0)

    # Compute Metrics
    accuracy = np.mean(np.array(y_true) == np.array(y_pred)) * 100
    precision = precision_score(y_true, y_pred, zero_division=1) * 100
    recall = recall_score(y_true, y_pred, zero_division=1) * 100
    f1 = f1_score(y_true, y_pred, zero_division=1) * 100

    print(f"\n📊 **Model Evaluation Metrics:**")
    print(f"✅ Accuracy: {accuracy:.2f}%")
    print(f"✅ Precision: {precision:.2f}%")
    print(f"✅ Recall: {recall:.2f}%")
    print(f"✅ F1-Score: {f1:.2f}%")

    return accuracy

# 🔹 **Plot reconstruction error distributions**
def plot_error_distributions(results, threshold):
    freshapple_errors = [r["Reconstruction Error"] for r in results if r["Folder"].lower() == "freshapples"]
    rottenapple_errors = [r["Reconstruction Error"] for r in results if r["Folder"].lower() == "rottenapples"]

    plt.figure(figsize=(10, 6))
    plt.hist(freshapple_errors, bins=30, alpha=0.7, label="Fresh Apples (Normal)")
    plt.hist(rottenapple_errors, bins=30, alpha=0.7, label="Rotten Apples (Anomalous)")

    # 🔥 **Mark normal and anomaly zones**
    plt.axvline(threshold, color="red", linestyle="dashed", linewidth=2, label="Threshold")
    plt.axvspan(0, threshold, color="green", alpha=0.2, label="Normal Zone")
    plt.axvspan(threshold, max(freshapple_errors + rottenapple_errors), color="red", alpha=0.2, label="Anomaly Zone")

    plt.title("🔍 Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# 🔹 **Run Testing**
if __name__ == "__main__":
    test_dir = "C:/Users/usbag/Desktop/Anomaly Detector/test"
    image_size = (128, 128)

    # Test the model and get all reconstruction errors
    results, all_errors = test_model_on_directory(test_dir, image_size)

    # Automatically determine the best threshold
    threshold = determine_threshold(all_errors, percentile=85)
    print(f"\n🚀 **Adaptive Threshold Determined: {threshold:.6f}**")

    # Save results to a CSV file
    save_results_to_csv(results)

    # Analyze results
    calculate_metrics(results, threshold)
    plot_error_distributions(results, threshold)