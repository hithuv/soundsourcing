# -*- coding: utf-8 -*-
"""
train_sound_model.py

A complete, self-contained script to train a simple CNN for sound classification
on the UrbanSound8K dataset. Optimized to run locally on Apple Silicon (M-series chips).
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION & PARAMETERS ---
print("🎧 Sound Classification Model Training Script 🎧")

# Paths (as specified by user)
METADATA_PATH = "UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "UrbanSound8K/audio"
OUTPUT_DIR = "output"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Audio & Spectrogram Parameters
SR = 22050          # Sample Rate
DURATION = 4        # Duration of audio clips in seconds
SAMPLES = SR * DURATION
N_MELS = 128        # Number of Mel bands
HOP_LENGTH = 512    # Hop length for STFT
N_FFT = 2048        # Number of FFT components

# Model Training Parameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# --- 2. DEVICE CONFIGURATION (for Apple Silicon) ---
# Check for Apple's Metal Performance Shaders (MPS) and set it as the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\n✅ MPS device found. Training will run on your Mac's GPU.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("\n✅ CUDA device found. Training will run on your NVIDIA GPU.")
else:
    device = torch.device("cpu")
    print("\n⚠️ No GPU accelerator found. Training will run on the CPU (will be slower).")

# --- 3. DATA PREPROCESSING FUNCTION ---
def preprocess_audio(file_path):
    """
    Loads an audio file, pads/truncates it to a fixed length, and computes its
    log-mel spectrogram.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=SR)

        # Pad or truncate to the fixed sample length
        if len(y) < SAMPLES:
            y = np.pad(y, (0, SAMPLES - len(y)), mode='constant')
        else:
            y = y[:SAMPLES]

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        
        # Convert to log scale (decibels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        return log_mel_spec
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- 4. LOADING & PROCESSING THE DATASET ---
# print("\n🔍 Loading metadata and processing audio files...")
# metadata = pd.read_csv(METADATA_PATH)

# features = []
# labels = []

# # Loop through the dataset with a progress bar
# for index, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing files"):
#     file_path = os.path.join(AUDIO_DIR, f"fold{row['fold']}", row['slice_file_name'])
    
#     # Check if file exists to prevent errors
#     if not os.path.exists(file_path):
#         print(f"File not found: {file_path}. Skipping.")
#         continue

#     # Preprocess the audio file to get the spectrogram
#     spectrogram = preprocess_audio(file_path)
    
#     if spectrogram is not None:
#         features.append(spectrogram)
#         labels.append(row['class'])

# print(f"\n✅ Processed {len(features)} audio files successfully.")

# # --- 5. DATA PREPARATION FOR PYTORCH ---
# print("\n⚙️ Preparing data for model training...")

# # Convert lists to NumPy arrays
# X = np.array(features)
# y = np.array(labels)

# # Encode string labels to integers
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# num_classes = len(le.classes_)
# print(f"Found {num_classes} classes: {list(le.classes_)}")

# # Add a channel dimension for the CNN (batch, channels, height, width)
# X = X[:, np.newaxis, :, :]

# # Split data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
# )

# # Convert NumPy arrays to PyTorch tensors
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)
# X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
# y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# # Create PyTorch Datasets and DataLoaders
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. DATA LOADING & PREPROCESSING ---

print("\n🔍 Checking for preprocessed data...")
# Define paths for the saved NumPy arrays
FEATURES_PATH = os.path.join(OUTPUT_DIR, "processed_features.npy")
LABELS_PATH = os.path.join(OUTPUT_DIR, "processed_labels.npy")

# Check if the preprocessed files already exist
if os.path.exists(FEATURES_PATH) and os.path.exists(LABELS_PATH):
    print("✅ Preprocessed data found! Loading from disk...")
    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH, allow_pickle=True) # allow_pickle for string array
    print(f"Loaded {len(X)} features and labels.")

else:
    # If data doesn't exist, run the original processing loop
    print("⚠️ No preprocessed data found. Processing audio files from scratch...")
    metadata = pd.read_csv(METADATA_PATH)
    features, labels = [], []
    for index, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing files"):
        file_path = os.path.join(AUDIO_DIR, f"fold{row['fold']}", row['slice_file_name'])
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping.")
            continue
        spectrogram = preprocess_audio(file_path)
        if spectrogram is not None:
            features.append(spectrogram)
            labels.append(row['class'])
    
    # Convert lists to NumPy arrays
    X = np.array(features)
    y = np.array(labels)

    print(f"\n✅ Processed {len(X)} audio files successfully.")
    
    # Save the processed data to disk for the next time
    print(f"💾 Saving processed data to disk...")
    np.save(FEATURES_PATH, X)
    np.save(LABELS_PATH, y)
    print(f"Data saved to {FEATURES_PATH} and {LABELS_PATH}")


# --- 5. DATA PREPARATION FOR PYTORCH ---
# This section now starts with the assumption that X and y are already loaded or created.
print("\n⚙️ Preparing data for model training...")

# Encode string labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"Found {num_classes} classes: {list(le.classes_)}")

# Add a channel dimension for the CNN (batch, channels, height, width)
X = X[:, np.newaxis, :, :]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Create PyTorch Datasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 6. MODEL DEFINITION (Simple 2D CNN) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Convolutional blocks (these are unchanged)
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # *** THE FIX IS HERE ***
        # Create a dummy input tensor to automatically calculate the flattened size.
        # This makes the model robust to changes in input dimensions.
        # We determine the number of time frames based on our audio parameters.
        time_frames = (SAMPLES // HOP_LENGTH) + 1
        dummy_input = torch.randn(1, 1, N_MELS, time_frames)
        dummy_output = self.conv_blocks(dummy_input)
        flattened_size = self.flatten(dummy_output).shape[1]
        
        # Fully Connected (Linear) Layer, now with the correct input size
        self.fc = nn.Linear(flattened_size, num_classes)

    def forward(self, x):
        # Pass input through the convolutional blocks
        x = self.conv_blocks(x)
        # Flatten the output for the linear layer
        x = self.flatten(x)
        # Pass through the final fully connected layer
        x = self.fc(x)
        return x
# Instantiate the model and move it to the configured device
model = SimpleCNN(num_classes).to(device)
print("\n📄 Model Summary:")
print(model)

# --- 7. TRAINING & VALIDATION LOOP ---
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\n🚀 Starting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    # Use tqdm for a progress bar on the training loader
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for inputs, labels in train_pbar:
        # Move data to the device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        # Update progress bar description
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct_train / total_train

    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad(): # No need to calculate gradients during validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = correct_val / total_val
    
    print(
        f"Epoch {epoch+1}/{EPOCHS} -> "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

print("\n🎉 Training finished!")

# --- 8. SAVE THE MODEL AND LABEL ENCODER ---
model_save_path = os.path.join(OUTPUT_DIR, "sound_cnn_model.pth")
le_save_path = os.path.join(OUTPUT_DIR, "label_encoder.npy")

torch.save(model.state_dict(), model_save_path)
np.save(le_save_path, le.classes_)

print(f"\n💾 Model saved to: {model_save_path}")
print(f"💾 Label encoder classes saved to: {le_save_path}")
print("\n✅ All done!")