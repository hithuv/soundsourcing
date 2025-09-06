# -*- coding: utf-8 -*-
"""
predict_sound.py (Updated)

A script to load the trained model and predict the class of an audio file
from the UrbanSound8K dataset, comparing it to the actual class.
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn

# --- 1. CONFIGURATION ---
print("🎤 Sound Classification Prediction & Verification Script 🎤")

# --- CHANGE THIS ---
# Relative path of the file from within the 'UrbanSound8K/audio/' folder.
# Example files you can try:
# "fold5/100032-3-0-0.wav"  (should be 'dog_bark')
# "fold1/101415-3-0-2.wav"  (should be 'dog_bark')
# "fold10/104817-4-0-0.wav" (should be 'drilling')
# "fold2/107853-5-0-0.wav"  (should be 'engine_idling')
AUDIO_TO_TEST_RELATIVE = "UrbanSound8K/audio/fold10/165166-8-0-5.wav"

# Paths to model, labels, and metadata
OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "sound_cnn_model.pth")
LE_PATH = os.path.join(OUTPUT_DIR, "label_encoder.npy")
METADATA_PATH = "UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_ROOT = "UrbanSound8K/audio"
AUDIO_ROOT = ""

# Audio parameters (MUST match the parameters from training)
SR = 22050
DURATION = 4
SAMPLES = SR * DURATION
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# --- 2. SETUP DEVICE, MODEL CLASS, AND HELPER FUNCTIONS ---
# Check for MPS device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\n✅ MPS device found. Using GPU for inference.")
else:
    device = torch.device("cpu")
    print("\n⚠️ No GPU found. Using CPU for inference.")

# Re-define the exact same model architecture used for training
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        time_frames = (SAMPLES // HOP_LENGTH) + 1
        dummy_input = torch.randn(1, 1, N_MELS, time_frames)
        dummy_output = self.conv_blocks(dummy_input)
        flattened_size = self.flatten(dummy_output).shape[1]
        self.fc = nn.Linear(flattened_size, num_classes)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Define the same preprocessing function used in training
def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR)
        if len(y) < SAMPLES: y = np.pad(y, (0, SAMPLES - len(y)))
        else: y = y[:SAMPLES]
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        return librosa.power_to_db(mel_spec, ref=np.max)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def get_actual_class(relative_path, metadata_df):
    """Looks up the true class label from the metadata DataFrame."""
    filename = os.path.basename(relative_path)
    file_metadata = metadata_df[metadata_df['slice_file_name'] == filename]
    if not file_metadata.empty:
        return file_metadata.iloc[0]['class']
    return "Unknown (file not in metadata)"


# --- 3. LOAD MODEL AND MAKE PREDICTION ---
def predict(audio_path, model, classes):
    """Predicts the sound class for a single audio file."""
    model.eval()
    spectrogram = preprocess_audio(audio_path)
    if spectrogram is None: return None, None

    spec_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    spec_tensor = spec_tensor.to(device)
    
    with torch.no_grad():
        output = model(spec_tensor)
        
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = classes[predicted_idx.item()]
    confidence_percent = confidence.item() * 100
    
    return predicted_class, confidence_percent

# Main execution block
if __name__ == "__main__":
    # Construct the full path to the audio file
    full_audio_path = os.path.join(AUDIO_ROOT, AUDIO_TO_TEST_RELATIVE)

    if not os.path.exists(full_audio_path):
        print(f"❌ Error: Audio file not found at '{full_audio_path}'")
        print("Please check the AUDIO_TO_TEST_RELATIVE variable.")
    else:
        # Load metadata, label encoder, and the trained model
        metadata = pd.read_csv(METADATA_PATH)
        le_classes = np.load(LE_PATH, allow_pickle=True)
        num_classes = len(le_classes)
        
        model = SimpleCNN(num_classes)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        
        # Get the actual class from the metadata
        actual_class = get_actual_class(AUDIO_TO_TEST_RELATIVE, metadata)

        # Get the model's prediction
        predicted_class, confidence = predict(full_audio_path, model, le_classes)
        
        # Print the final comparison
        print("\n-------------------------------------------")
        print(f"🔊 Analysis for: {AUDIO_TO_TEST_RELATIVE}")
        print(f"    - ✅ Actual Class:    '{actual_class}'")
        if predicted_class:
            print(f"    - 🤖 Predicted Class: '{predicted_class}' (Confidence: {confidence:.2f}%)")
        else:
            print("    - 🤖 Prediction failed.")
        print("-------------------------------------------")