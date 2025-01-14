import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load the trained model
model_path = "improved_autoencoder.keras"  # Update to match the trained model file
autoencoder = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")


# Preprocess a single test image
def preprocess_test_image(image_path, image_size=(128, 128)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = image / 255.0  # Normalize
    return tf.expand_dims(image, axis=0)  # Add batch dimension


# Calculate reconstruction error
def calculate_reconstruction_error(original, reconstructed):
    return np.mean(np.square(original - reconstructed))


# Test the autoencoder on a directory of images
def test_model_on_directory(base_dir, threshold, image_size=(128, 128)):
    results = []
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        print(f"\nTesting images in folder: {class_name}")

        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Preprocess the image
            test_image = preprocess_test_image(image_path, image_size)

            # Get the reconstruction from the model
            reconstructed_image = autoencoder.predict(test_image)
            reconstruction_error = calculate_reconstruction_error(test_image, reconstructed_image)

            # Classify the image
            classification = "Normal" if reconstruction_error <= threshold else "Anomalous"

            results.append({
                "Image": filename,
                "Folder": class_name,
                "Reconstruction Error": reconstruction_error,
                "Classification": classification
            })

    return results


# Save results to a CSV file
def save_results_to_csv(results, output_file="test_results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


# Calculate Accuracy
def calculate_accuracy(results):
    correct = 0
    total = len(results)

    for result in results:
        ground_truth = "Normal" if result["Folder"].lower() == "freshtomato" else "Anomalous"
        print(f"Image: {result['Image']}, Ground Truth: {ground_truth}, Classification: {result['Classification']}")

        # Compare the ground truth with the classification
        if result["Classification"] == ground_truth:
            correct += 1

    # Handle case when total is zero (to prevent division by zero)
    if total == 0:
        print("\nNo test images found. Accuracy cannot be calculated.")
        return 0.0

    # Calculate accuracy as a percentage
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total} correctly classified)")
    return accuracy

def visualize_reconstruction_errors(results):
    freshtomato_errors = [r["Reconstruction Error"] for r in results if r["Folder"].lower() == "freshtomato"]
    rottentomato_errors = [r["Reconstruction Error"] for r in results if r["Folder"].lower() == "rottentomato"]

    plt.figure(figsize=(10, 6))
    plt.hist(freshtomato_errors, bins=30, alpha=0.7, label="Freshtomato (Normal)")
    plt.hist(rottentomato_errors, bins=30, alpha=0.7, label="Rottentomato (Anomalous)")
    plt.axvline(x=0.002, color='r', linestyle='dashed', linewidth=2, label="Threshold")
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# Plot reconstruction error distributions
def plot_error_distributions(results, threshold):
    freshtomato_errors = [r["Reconstruction Error"] for r in results if r["Folder"].lower() == "freshtomato"]
    rottentamto_errors = [r["Reconstruction Error"] for r in results if r["Folder"].lower() == "rottentamto"]

    plt.figure(figsize=(10, 6))
    plt.hist(freshtomato_errors, bins=20, alpha=0.7, label="Freshtomato (Normal)")
    plt.hist(rottentamto_errors, bins=20, alpha=0.7, label="Rottentamto (Anomalous)")
    plt.axvline(threshold, color="red", linestyle="dashed", linewidth=2, label="Threshold")
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# Main function to test the autoencoder
def main():
    # Define paths and parameters
    test_dir = "C:\\Users\\usbag\\Desktop\\cap\\test"      # Test data directory
    threshold = 0.004207843                                     # Threshold for anomaly detection
    image_size = (128, 128)                               # Image dimensions

    # Test the model
    results = test_model_on_directory(test_dir, threshold, image_size)

    # Save results to a CSV file
    save_results_to_csv(results)

    # Analyze results
    calculate_accuracy(results)
    plot_error_distributions(results, threshold)
    
    visualize_reconstruction_errors(results)



if __name__ == "__main__":
    main()
