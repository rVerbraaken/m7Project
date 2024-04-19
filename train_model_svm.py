#Colin Harmsen & Roel Verbraaken
#Based on excersises and information on canvas, Programming and Hands-on AI, Creative Technology Module 7

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

# Define edge detection kernel
edge_detection_kernel = np.array([[-1, -1, -1],
                                  [-1, 8, -1],
                                  [-1, -1, -1]])

#Ensuring that the random number generation is consistent across runs
keras.utils.set_random_seed(7)

# Function to preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (100, 100))
    custom_convolution = cv2.filter2D(resized_img, -1, edge_detection_kernel)
    return custom_convolution

# Function to load images and skeleton data from folders
def load_images_and_data(root_folder):
    images = []
    labels = []
    folder_count = 0
    total_images = 0

    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            folder_count += 1
            print("Loading images from folder:", folder)
            image_count = 0
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    total_images += 1
                    image_count += 1
                    print("\rProcessed {} images in folder {}".format(image_count, folder), end="")
                    processed_image = preprocess_image(img_path)
                    images.append(processed_image)
                    labels.append(folder)
    print("\nLoaded images from", folder_count, "folders with a total of", total_images, "images.")
    return images, labels

# Load images and skeleton data from folders
images, labels= load_images_and_data(r"mother_folder")

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Flatten images
images_flat = np.array([img.flatten() for img in images])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm = SVC(kernel='linear')
print("Training SVM classifier...")
svm.fit(X_train, y_train)
print("SVM training completed.")

# Define parameter grid for GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 10], 'gamma': [10, 1, 0.1, 0.01], 'kernel': ['linear']}

# Perform grid search cross-validation
print("Performing GridSearchCV for hyperparameter tuning...")
grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)
print("GridSearchCV completed.")

best_params = grid_search.best_params_
print("Best parameters found:", best_params)

best_model = grid_search.best_estimator_

y_pred = grid_search.predict(X_test)

# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix, reference chatgpt, prompt: plot a confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Plot confusion matrix for test set
plot_confusion_matrix(y_test, y_pred, labels=np.unique(labels))

# Save best model to files
dump(best_model, 'best_model.joblib')
print("Best model saved to 'best_model.joblib'.")