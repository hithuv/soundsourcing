import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import librosa

# --- CONFIGURATION ---
print("🧠 Training Multi-Label Sound Event Detection Model 🧠")
DATA_DIR = "UrbanSound8K_mixed"
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
OUTPUT_DIR = "output"

# Audio parameters (must match generation script)
SR = 22050
DURATION = 4
SAMPLES = SR * DURATION
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
NUM_CLASSES = 10 # From UrbanSound8K

# Training parameters
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

# --- SETUP DEVICE & MODEL ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nUsing device: {device}")

# The model architecture is the same, just the final output interpretation changes
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
        # We return raw logits; the loss function will handle the sigmoid
        x = self.fc(x)
        return x

# --- DATASET & DATALOADER ---
class MixedSoundDataset(Dataset):
    def __init__(self, df, audio_dir):
        self.df = df
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['filename'])
        
        # Load and preprocess audio to spectrogram
        y, sr = librosa.load(file_path, sr=SR)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Convert label string '1,0,1...' to a float tensor
        label = np.array([float(x) for x in row['classes'].split(',')])
        
        return torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.float32)

metadata = pd.read_csv(METADATA_PATH)
train_df, val_df = train_test_split(metadata, test_size=0.2, random_state=42)

train_dataset = MixedSoundDataset(train_df, AUDIO_DIR)
val_dataset = MixedSoundDataset(val_df, AUDIO_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- TRAINING LOOP ---
model = SimpleCNN(NUM_CLASSES).to(device)
# *** KEY CHANGE: Use BCEWithLogitsLoss for multi-label classification ***
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\n🚀 Starting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    train_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    correct_preds, total_preds = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy: apply sigmoid and threshold at 0.5
            preds = torch.sigmoid(outputs) > 0.5
            total_preds += labels.numel()
            correct_preds += (preds == labels.bool()).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = correct_preds / total_preds
    
    print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Save the model
model_save_path = os.path.join(OUTPUT_DIR, "multi_label_cnn_model.pth")
torch.save(model.state_dict(), model_save_path)
print(f"\n✅ Training finished! Model saved to '{model_save_path}'")