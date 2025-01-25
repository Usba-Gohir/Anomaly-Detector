import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LeakyReLU, Dropout


# Define the improved autoencoder architecture
def build_autoencoder(input_shape):
    # Encoder
    input_img = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = LeakyReLU(alpha=0.1)(x)  # LeakyReLU instead of ReLU
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)  # Add dropout to prevent overfitting

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    # Bottleneck
    encoded = Conv2D(16, (3, 3), padding='same')(x)
    encoded = LeakyReLU(alpha=0.1)(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)  # Add another bottleneck layer

    # Decoder
    x = Conv2D(16, (3, 3), padding='same')(encoded)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    x = Dropout(0.3)(x)  # Dropout in the decoder

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder


# Load and preprocess images from a directory
def load_images_from_directory(directory, image_size=(128, 128)):
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            image = tf.io.read_file(img_path)
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.resize(image, image_size[:2])
            image = image / 255.0  # Normalize to [0, 1]
            images.append(image)
    return tf.stack(images)


# Main function to train the autoencoder
def main():
    # Define paths
    train_dir = "C:\\Users\\usbag\\Desktop\\Anomaly Detector\\freshtomato"  # Training data directory
    output_model_path = "improved_autoencoder.keras"           # Save trained model as `.keras`
    image_size = (128, 128, 3)                                 # Image dimensions

    # Load training data
    train_images = load_images_from_directory(train_dir)
    print(f"Loaded {len(train_images)} training images from {train_dir}")

    # Build and compile the autoencoder
    autoencoder = build_autoencoder(image_size)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Define the checkpoint to save the best model
    checkpoint = ModelCheckpoint(
        output_model_path,  # Path to save the model
        monitor="loss", 
        save_best_only=True, 
        mode="min", 
        verbose=1
    )

    # Train the autoencoder
    print("Starting training...")
    autoencoder.fit(
        train_images,  # Input images
        train_images,  # Target is the same as the input (autoencoder)
        epochs=50,     # Number of epochs
        batch_size=16, # Batch size
        callbacks=[checkpoint],  # Save the best model
        shuffle=True
    )

    print(f"Training completed. Model saved to {output_model_path}")


if __name__ == "__main__":
    main()
