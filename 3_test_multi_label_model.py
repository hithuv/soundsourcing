# -*- coding: utf-8 -*-
"""
3_test_multi_label_model.py

A script to test the trained multi-label sound event detection model.
It loads a mixed audio file, predicts which sounds are present, and
compares the prediction to the actual ground truth.
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn

# --- 1. CONFIGURATION ---
print("🎤 Multi-Label Sound Event Detection - Test Script 🎤")

# --- CHANGE THIS ---
# Path to the mixed audio file you want to test.
AUDIO_TO_TEST = "output_separated.wav"

# --- FIXED PATHS & PARAMETERS ---
OUTPUT_DIR = "output"
DATA_DIR = "UrbanSound8K_mixed"
MODEL_PATH = os.path.join(OUTPUT_DIR, "multi_label_cnn_model.pth")
LE_PATH = os.path.join(OUTPUT_DIR, "label_encoder.npy") # For class names
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

# Audio parameters (MUST match training script)
SR = 22050
DURATION = 4
SAMPLES = SR * DURATION
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
NUM_CLASSES = 10

# Prediction threshold: A sound is considered "present" if its probability is above this value.
PREDICTION_THRESHOLD = 0.5

# --- 2. SETUP DEVICE, MODEL, AND HELPERS ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Re-define the exact same model architecture
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

# Preprocessing function
def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        return librosa.power_to_db(mel_spec, ref=np.max)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- 3. MAIN PREDICTION BLOCK ---
if __name__ == "__main__":
    if not os.path.exists(AUDIO_TO_TEST):
        print(f"❌ Error: Audio file not found at '{AUDIO_TO_TEST}'")
    else:
        # Load metadata, class names, and the trained model
        metadata = pd.read_csv(METADATA_PATH)
        class_names = np.load(LE_PATH, allow_pickle=True)
        
        model = SimpleCNN(NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()

        # --- Get Actual Sounds ---
        filename = os.path.basename(AUDIO_TO_TEST)
        file_meta = metadata[metadata['filename'] == filename]
        
        if file_meta.empty:
            print(f"Could not find metadata for {filename}")
            actual_sounds = []
        else:
            actual_labels_vector = np.array([int(x) for x in file_meta.iloc[0]['classes'].split(',')])
            actual_sounds = [class_names[i] for i, present in enumerate(actual_labels_vector) if present == 1]

        # --- Get Predicted Sounds ---
        spectrogram = preprocess_audio(AUDIO_TO_TEST)
        spec_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(spec_tensor)
            # Apply sigmoid to convert logits to independent probabilities
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        predicted_sounds = []
        confidences = []
        for i, prob in enumerate(probabilities):
            if prob > PREDICTION_THRESHOLD:
                predicted_sounds.append(class_names[i])
                confidences.append(prob)

        # --- Display Results ---
        print("\n" + "="*50)
        print(f"ANALYSIS FOR: {filename}")
        print("="*50)
        
        print("\n✅ ACTUAL SOUNDS:")
        if actual_sounds:
            for sound in sorted(actual_sounds):
                print(f"  - {sound}")
        else:
            print("  - (None found in metadata)")
            
        print("\n🤖 PREDICTED SOUNDS:")
        if predicted_sounds:
            for sound, conf in sorted(zip(predicted_sounds, confidences)):
                print(f"  - {sound} (Confidence: {conf*100:.2f}%)")
        else:
            print("  - (None detected)")
        print("\n" + "="*50)