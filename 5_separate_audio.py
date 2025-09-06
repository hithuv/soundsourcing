import os
import numpy as np
import torch
import librosa
import soundfile as sf
# Make sure you have renamed the training script to 'train_separation_model.py'
from train_separation_model_4 import UNet

print("✨ Audio Separation Script ✨")

# --- 1. CONFIGURATION ---

# --- EDIT THESE TWO LINES ---
MIXED_AUDIO_PATH = "UrbanSound8K_separation/audio/mixed_sep_0005.wav"
CLASS_TO_REMOVE = "siren"
# ---------------------------

OUTPUT_FILENAME = "output_separated.wav"

# --- FIXED PATHS & PARAMETERS ---
OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "separation_unet_model.pth")
LE_PATH = os.path.join(OUTPUT_DIR, "label_encoder.npy")
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
NUM_CLASSES = 10

# --- 2. SETUP ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nUsing device: {device}")

class_names = np.load(LE_PATH, allow_pickle=True)
class_to_idx = {name: i for i, name in enumerate(class_names)}

if CLASS_TO_REMOVE not in class_to_idx:
    raise ValueError(f"Error: Class '{CLASS_TO_REMOVE}' not found. Please choose from: {class_names}")

model = UNet(n_class=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- 3. SEPARATION PROCESS ---
try:
    print(f"Loading mixed audio: {MIXED_AUDIO_PATH}")
    y_mix, _ = librosa.load(MIXED_AUDIO_PATH, sr=SR)
    
    s_mix_full = librosa.stft(y_mix, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag_mix, phase_mix = librosa.magphase(s_mix_full)
    
    mag_mix_tensor = torch.tensor(mag_mix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    print("Generating source masks...")
    with torch.no_grad():
        predicted_masks = model(mag_mix_tensor)

    remove_idx = class_to_idx[CLASS_TO_REMOVE]
    print(f"Applying inverted mask for '{CLASS_TO_REMOVE}'...")
    mask_to_remove = predicted_masks[0, remove_idx, :, :].cpu().numpy()
    
    final_mask = 1 - mask_to_remove
    
    # *** THIS IS THE FIX ***
    # Crop the original magnitude and phase to match the mask's (model's output) size.
    target_H, target_W = final_mask.shape
    mag_mix_cropped = mag_mix[:target_H, :target_W]
    phase_mix_cropped = phase_mix[:target_H, :target_W]
    # ***********************

    # Apply the mask to the CROPPED original magnitude
    mag_final = mag_mix_cropped * final_mask
    
    print("Reconstructing audio from the modified spectrogram...")
    # Reconstruct using the new magnitude and the CROPPED original phase
    y_final = librosa.istft(mag_final * phase_mix_cropped, hop_length=HOP_LENGTH)
    
    sf.write(OUTPUT_FILENAME, y_final, SR)
    print(f"\n✅ Separation complete! Audio saved to '{OUTPUT_FILENAME}'")

except Exception as e:
    print(f"\nAn error occurred: {e}")