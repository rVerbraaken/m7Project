#Roel Verbraaken & Colin Harmsen
#program trains ai based on the skeleton data that was gathered at the same time as the pictures

import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import accuracy_score

# Load the JSON file
with open('data.json', 'r') as f:
    data = json.load(f)

# Extract features and original labels
X = np.array([np.array(d['skeleton_data']) for d in data])
original_labels = np.array([d['label'] for d in data])

# Map original labels to common labels
common_labels = np.array([label[0] for label in original_labels])
# Use label encoding to convert string labels to integer labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(common_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(np.unique(encoded_labels)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)