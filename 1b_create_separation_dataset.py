import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🔊 Creating a Dataset for Audio Separation 🔊")

# --- CONFIGURATION ---
INPUT_METADATA = "UrbanSound8K/metadata/UrbanSound8K.csv"
INPUT_AUDIO_DIR = "UrbanSound8K/audio"
OUTPUT_DIR = "UrbanSound8K_separation"
OUTPUT_AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")

NUM_SYNTHETIC_FILES = 2000
NUM_SOURCES_MIN = 2
NUM_SOURCES_MAX = 3
SR = 22050
DURATION = 4
SAMPLES = SR * DURATION

os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
metadata = pd.read_csv(INPUT_METADATA)
new_metadata = []

for i in tqdm(range(NUM_SYNTHETIC_FILES), desc="Generating mixed audio"):
    num_sources = np.random.randint(NUM_SOURCES_MIN, NUM_SOURCES_MAX + 1)
    selected_rows = metadata.groupby('classID').sample(1).sample(n=num_sources)
    
    mixed_audio = np.zeros(SAMPLES, dtype=np.float32)
    source_paths = []
    
    for _, row in selected_rows.iterrows():
        file_path = os.path.join(INPUT_AUDIO_DIR, f"fold{row['fold']}", row['slice_file_name'])
        y, _ = librosa.load(file_path, sr=SR)
        if len(y) < SAMPLES: y = np.pad(y, (0, SAMPLES - len(y)))
        else: y = y[:SAMPLES]
        
        y /= np.max(np.abs(y)) + 1e-8
        mixed_audio += y
        source_paths.append(file_path) # Store the path to the clean source
            
    mixed_audio /= np.max(np.abs(mixed_audio)) + 1e-8
    
    output_filename = f"mixed_sep_{i:04d}.wav"
    output_path = os.path.join(OUTPUT_AUDIO_DIR, output_filename)
    sf.write(output_path, mixed_audio, SR)
    
    new_metadata.append({
        "mix_filename": output_filename,
        "source_paths": ';'.join(source_paths) # Use a separator for the paths
    })

df_new = pd.DataFrame(new_metadata)
df_new.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

print(f"\n✅ Successfully created separation dataset in '{OUTPUT_DIR}'")