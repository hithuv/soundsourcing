import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

# --- CONFIGURATION ---
print("🔊 Creating Multi-Source Audio Dataset 🔊")
INPUT_METADATA = "UrbanSound8K/metadata/UrbanSound8K.csv"
INPUT_AUDIO_DIR = "UrbanSound8K/audio"
OUTPUT_DIR = "UrbanSound8K_mixed"
OUTPUT_AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")

NUM_SYNTHETIC_FILES = 2000 # Number of mixed audio files to generate
NUM_SOURCES_MIN = 2        # Mix at least 2 sounds
NUM_SOURCES_MAX = 3        # Mix at most 3 sounds

# Audio parameters
SR = 22050
DURATION = 4
SAMPLES = SR * DURATION

# Create output directories
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

# --- SCRIPT ---
metadata = pd.read_csv(INPUT_METADATA)
# Get a list of unique classes to create multi-hot labels later
class_list = sorted(metadata['class'].unique())
class_to_int = {cls: i for i, cls in enumerate(class_list)}

new_metadata = []

for i in tqdm(range(NUM_SYNTHETIC_FILES), desc="Generating mixed audio"):
    num_sources = np.random.randint(NUM_SOURCES_MIN, NUM_SOURCES_MAX + 1)
    
    # Select unique files from different classes
    selected_rows = metadata.groupby('classID').sample(1)
    selected_rows = selected_rows.sample(n=num_sources)
    
    mixed_audio = np.zeros(SAMPLES, dtype=np.float32)
    present_classes = set()
    
    for _, row in selected_rows.iterrows():
        try:
            # Load and preprocess audio
            file_path = os.path.join(INPUT_AUDIO_DIR, f"fold{row['fold']}", row['slice_file_name'])
            y, sr = librosa.load(file_path, sr=SR)
            
            # Pad or truncate
            if len(y) < SAMPLES:
                y = np.pad(y, (0, SAMPLES - len(y)))
            else:
                y = y[:SAMPLES]
                
            # Normalize volume before mixing
            y /= np.max(np.abs(y)) + 1e-8
            
            mixed_audio += y
            present_classes.add(row['class'])
            
        except Exception as e:
            print(f"Skipping file {row['slice_file_name']} due to error: {e}")
            continue
            
    # Normalize the final mixed audio to prevent clipping
    mixed_audio /= np.max(np.abs(mixed_audio)) + 1e-8
    
    # Create multi-hot encoded label
    label_vector = [0] * len(class_list)
    for cls in present_classes:
        label_vector[class_to_int[cls]] = 1
        
    # Save audio and metadata
    output_filename = f"mixed_{i:04d}.wav"
    output_path = os.path.join(OUTPUT_AUDIO_DIR, output_filename)
    sf.write(output_path, mixed_audio, SR)
    
    # The label vector is stored as a string for easy CSV saving
    new_metadata.append({
        "filename": output_filename,
        "classes": ','.join(map(str, label_vector))
    })

# Save the new metadata file
df_new = pd.DataFrame(new_metadata)
df_new.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

print(f"\n✅ Successfully created {NUM_SYNTHETIC_FILES} mixed audio files in '{OUTPUT_DIR}'")