import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import librosa



# --- CONFIGURATION ---
DATA_DIR = "UrbanSound8K_separation"
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
ORIGINAL_METADATA_PATH = "UrbanSound8K/metadata/UrbanSound8K.csv"
OUTPUT_DIR = "output"

SR = 22050
DURATION = 4
SAMPLES = SR * DURATION
N_FFT = 2048
HOP_LENGTH = 512
NUM_CLASSES = 10

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.001

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nUsing device: {device}")

le_classes = np.load(os.path.join(OUTPUT_DIR, "label_encoder.npy"), allow_pickle=True)
class_to_idx = {name: i for i, name in enumerate(le_classes)}

# --- U-NET MODEL DEFINITION (WITH BATCH NORMALIZATION) ---
# Helper block for building the U-Net
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        # Encoder (downsampling path)
        self.enc1 = conv_block(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = conv_block(32, 64)
        
        # Decoder (upsampling path)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = conv_block(32, 16)
        
        # Final output layer
        self.outconv = nn.Conv2d(16, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder
        u1 = self.upconv1(b)
        # Crop and concatenate skip connection 1
        e2_cropped = e2[:, :, :u1.size()[2], :u1.size()[3]]
        cat1 = torch.cat([u1, e2_cropped], dim=1)
        d1 = self.dec1(cat1)
        
        u2 = self.upconv2(d1)
        # Crop and concatenate skip connection 2
        e1_cropped = e1[:, :, :u2.size()[2], :u2.size()[3]]
        cat2 = torch.cat([u2, e1_cropped], dim=1)
        d2 = self.dec2(cat2)
        
        # Output
        out = torch.sigmoid(self.outconv(d2))
        return out

# --- DATASET CLASS ---
# (This class remains the same as the previous version)
class SeparationDataset(Dataset):
    def __init__(self, df, mix_dir, class_map, original_metadata):
        self.df = df
        self.mix_dir = mix_dir
        self.class_map = class_map
        self.original_metadata = original_metadata

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mix_path = os.path.join(self.mix_dir, row['mix_filename'])
        y_mix, _ = librosa.load(mix_path, sr=SR)
        s_mix = librosa.stft(y_mix, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mag_mix, _ = librosa.magphase(s_mix)
        
        target_mags = torch.zeros((NUM_CLASSES, mag_mix.shape[0], mag_mix.shape[1]), dtype=torch.float32)

        for source_path in row['source_paths'].split(';'):
            fname = os.path.basename(source_path)
            class_name = self.original_metadata[self.original_metadata['slice_file_name'] == fname].iloc[0]['class']
            class_idx = self.class_map[class_name]
            
            y_source, _ = librosa.load(source_path, sr=SR)
            if len(y_source) < SAMPLES: y_source = np.pad(y_source, (0, SAMPLES - len(y_source)))
            else: y_source = y_source[:SAMPLES]

            s_source = librosa.stft(y_source, n_fft=N_FFT, hop_length=HOP_LENGTH)
            mag_source, _ = librosa.magphase(s_source)
            if mag_source.shape[1] != mag_mix.shape[1]:
                mag_source = librosa.util.fix_length(mag_source, size=mag_mix.shape[1], axis=1)

            target_mags[class_idx] = torch.tensor(mag_source, dtype=torch.float32)
            
        return torch.tensor(mag_mix, dtype=torch.float32).unsqueeze(0), target_mags

# --- TRAINING LOOP ---

def main():

    print("🧠 Training U-Net Model for Sound Separation (Corrected) 🧠")

    metadata = pd.read_csv(METADATA_PATH)
    original_metadata_df = pd.read_csv(ORIGINAL_METADATA_PATH)
    dataset = SeparationDataset(metadata, AUDIO_DIR, class_to_idx, original_metadata_df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = UNet(n_class=NUM_CLASSES).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for mix_mags, source_mags in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            mix_mags, source_mags = mix_mags.to(device), source_mags.to(device)
            optimizer.zero_grad()
            predicted_masks = model(mix_mags)
            
            target_H, target_W = predicted_masks.size()[2], predicted_masks.size()[3]
            mix_mags_cropped = mix_mags[:, :, :target_H, :target_W]
            source_mags_cropped = source_mags[:, :, :target_H, :target_W]

            reconstructed_mags = mix_mags_cropped * predicted_masks
            loss = criterion(reconstructed_mags, source_mags_cropped)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} -> Loss: {avg_loss:.4f}")

    model_save_path = os.path.join(OUTPUT_DIR, "separation_unet_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ Training finished! Model saved to '{model_save_path}'")

if __name__ == "__main__":
    main()