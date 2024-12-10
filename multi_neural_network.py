import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, ReLU, Flatten, Dense, Concatenate, Softmax
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Load the dataset
file_path = 'C:/Users/Dhruv/PycharmProjects/DeepLearning/autism_marker_EEG/combined_time_freq.csv'
data = pd.read_csv(file_path)
print(f"Initial number of participants: {data['participant_id'].nunique()}")

# Prepare the data: everything except participant_id, channel, and the target column
features = [col for col in data.columns if col not in ['participant_id', 'channel', 'Autistic']]
target = "Autistic"

# Identify participants with at least 6 channels
min_required_channels = 6
complete_participants = data.groupby('participant_id')['channel'].nunique()

# Filter participants with at least 6 channels
valid_participants = complete_participants[complete_participants >= min_required_channels].index
print(f"Participants with {min_required_channels} channels: {len(valid_participants)}")

# Filter the dataset to include only valid participants
data_filtered = data[data['participant_id'].isin(valid_participants)]
print(f"Number of participants after filtering: {data_filtered['participant_id'].nunique()}")

# Restrict to exactly 6 channels per participant
selected_channels = data_filtered['channel'].unique()[:min_required_channels]
data_filtered = data_filtered[data_filtered['channel'].isin(selected_channels)]
print(f"Remaining participants after selecting specific channels: {data_filtered['participant_id'].nunique()}")

# Split the data by channel for valid participants
X = {ch: data_filtered[data_filtered['channel'] == ch][features].values for ch in selected_channels}
y = data_filtered[data_filtered['channel'] == selected_channels[0]][
    target].values  # Target is the same for all channels of a participant

# Standardize features
scalers = {ch: StandardScaler().fit(X[ch]) for ch in X}
X_scaled = {ch: scalers[ch].transform(X[ch]) for ch in X}

# Align sample counts across channels
min_samples = min(X_scaled[ch].shape[0] for ch in selected_channels)
X_scaled = {ch: X_scaled[ch][:min_samples] for ch in selected_channels}
y = y[:min_samples]  # Adjust target size to match
print(f"Number of participants after aligning sample counts: {min_samples}")

# Combine channel features for each participant into a single 3D array (samples, channels, features)
X_combined = np.stack([X_scaled[ch] for ch in selected_channels],
                      axis=1)  # Shape: (num_samples, num_channels, num_features)
print(f"Shape of combined data: {X_combined.shape}")

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")


# Define the model
def create_model(input_shapes):
    inputs = []
    conv_outputs = []

    # Create convolutional layers for each channel
    for ch, input_shape in input_shapes.items():
        input_layer = Input(shape=input_shape, name=f"input_channel_{ch}")
        x = Conv1D(32, kernel_size=3, strides=1, padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv1D(64, kernel_size=3, strides=1, padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Flatten()(x)
        inputs.append(input_layer)
        conv_outputs.append(x)

    # Concatenate outputs from all channels
    concatenated = Concatenate()(conv_outputs)

    # Fully connected layers
    x = Dense(128, activation="relu")(concatenated)
    x = Dense(64, activation="relu")(x)
    output = Dense(2, activation="softmax", name="output")(x)

    # Define model
    model = Model(inputs=inputs, outputs=output)
    return model


# Define input shapes
input_shapes = {ch: (X_train.shape[2], 1) for ch in range(X_train.shape[1])}

# Instantiate the model
model = create_model(input_shapes)

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Prepare inputs for training
train_inputs = {f"input_channel_{i}": X_train[:, i, :].reshape(-1, X_train.shape[2], 1) for i in
                range(X_train.shape[1])}
test_inputs = {f"input_channel_{i}": X_test[:, i, :].reshape(-1, X_test.shape[2], 1) for i in range(X_test.shape[1])}

# Train the model
history = model.fit(train_inputs, y_train, validation_data=(test_inputs, y_test), epochs=50, batch_size=16)

# Evaluate the model
results = model.evaluate(test_inputs, y_test)
print("Test Loss, Test Accuracy:", results)

# Make predictions
y_pred = model.predict(test_inputs)

# Extract the probabilities for class 1 (the second column)
y_pred_class_1 = y_pred[:, 1]  # Probabilities for class 1

# Ensure predictions are of the correct shape
print(f"Shape of y_pred_class_1: {y_pred_class_1.shape}")

# Evaluate performance metrics
y_pred_binary = (y_pred_class_1 > 0.5).astype(int)  # Convert probabilities to binary labels

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))

# Print ROC AUC Score for class 1
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_class_1):.4f}")
