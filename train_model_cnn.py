import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Function to apply salt and pepper noise
def apply_salt_pepper(image, amount=0.01):
    row, col, _ = image.shape
    s_vs_p = 0.5
    out = np.copy(image)

    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords[0], coords[1], :] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords[0], coords[1], :] = 0
    return out

# Function to apply overexposure
def apply_overexposure(image, brightness_factor=1.2):
    # Increase brightness by multiplying pixel values by the brightness_factor
    bright_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return bright_image

# Function to apply underexposure
def apply_underexposure(image, brightness_factor=0.8):
    # Decrease brightness by multiplying pixel values by the brightness_factor
    dark_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return dark_image

dataset_path = "mother_folder"

# Initialize empty lists to store images and labels
X = []
y = []

# Loop through each folder in the mother folder
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Loop through each image in the folder
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)

            # Load the image
            image = cv2.imread(image_path)

            # Resize image to desired dimensions (e.g., 100x100)
            resized_img = cv2.resize(image, (100, 100))

            # Normalize pixel values to range [0, 1]
            normalized_img = resized_img / 255.0

            # Append the normalized image to X
            X.append(normalized_img)
            y.append(folder_name)  # Append the labels to y

            # Apply brightness adjustment
            if random.choice([True, False]):
                augmented_image = apply_overexposure(resized_img)
            else:
                augmented_image = apply_underexposure(resized_img)

            # Apply salt and pepper noise
            noisy_image = apply_salt_pepper(augmented_image)

            # Normalize pixel values to range [0, 1]
            normalized_noisy_img = noisy_image / 255.0

            # Append the normalized noisy image to X
            X.append(normalized_noisy_img)
            y.append(folder_name)  # Append the label (folder name) to y

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to numeric format
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Define model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Plot training history
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()

# Model Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print("Test Accuracy:", test_accuracy)

# F1 Score
y_pred = np.argmax(model.predict(X_test), axis=-1)
f1 = f1_score(y_test_encoded, y_pred, average='weighted')
print("F1 Score:", f1)

# Save the model
model.save("my_model_new.keras")

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test_encoded, y_pred)

# Plotting Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
