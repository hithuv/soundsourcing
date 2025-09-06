"""MINI2

"""

import os
import random
import pandas as pd
import librosa
import soundfile as sf
import numpy as np

# === Paths ===
metadata_path = ".UrbanSound8K/metadata/UrbanSound8K.csv"
audio_dir = "UrbanSound8K/audio"
output_dataset_dir = "output"

os.makedirs(output_dataset_dir, exist_ok=True)

# === Parameters ===
SR = 22050                # Sample rate
MIN_DURATION = 60         # minimum length = 60 sec
MAX_DURATION = 120        # maximum length (change as needed)
N_SYNTHETIC_FILES = 50    # number of synthetic long audios to generate

# === Load metadata ===
metadata = pd.read_csv(metadata_path)

# === Function to create one synthetic file ===
def create_synthetic_file(file_index, min_dur, max_dur):
    target_duration = random.randint(min_dur, max_dur)  # random duration between min & max

    current_time = 0.0
    long_audio = []
    timestamps = []

    # Shuffle dataset each time
    files = metadata.sample(frac=1).reset_index(drop=True)

    for idx, row in files.iterrows():
        fold = f"fold{row['fold']}"
        file_name = row['slice_file_name']
        label = row['class']
        file_path = os.path.join(audio_dir, fold, file_name)

        try:
            y, sr = librosa.load(file_path, sr=SR)
            duration = librosa.get_duration(y=y, sr=sr)

            if current_time + duration > target_duration:
                break

            # Append audio
            long_audio.append(y)

            # Add timestamp
            timestamps.append({
                "start_time": round(current_time, 2),
                "end_time": round(current_time + duration, 2),
                "label": label
            })

            current_time += duration

        except Exception as e:
            print(f"⚠️ Error with {file_path}: {e}")

    # Concatenate audio
    final_audio = np.concatenate(long_audio)

    # Save audio
    audio_out_path = os.path.join(output_dataset_dir, f"synthetic_{file_index}.wav")
    sf.write(audio_out_path, final_audio, SR)

    # Save timestamps
    timestamp_df = pd.DataFrame(timestamps)
    csv_out_path = os.path.join(output_dataset_dir, f"synthetic_{file_index}_timestamps.csv")
    timestamp_df.to_csv(csv_out_path, index=False)

    print(f"✅ Saved: {audio_out_path} (duration ~{target_duration}s) and {csv_out_path}")

# === Generate multiple synthetic files ===
for i in range(N_SYNTHETIC_FILES):
    create_synthetic_file(i, MIN_DURATION, MAX_DURATION)

import os
import librosa
import numpy as np
import pandas as pd

# === Paths ===
synthetic_dir = "/content/drive/MyDrive/CRNN_dataset"
save_X_path = "/content/drive/MyDrive/CRNN_X.npy"
save_y_path = "/content/drive/MyDrive/CRNN_y.npy"

# === Parameters ===
SR = 22050
N_MELS = 64
WINDOW_SIZE = 1.0   # 1 second window
HOP_SIZE = 0.5      # 50% overlap

X = []
y = []

# === Function: Extract Log-Mel Spectrogram ===
def extract_logmel(y_segment, sr, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel, ref=np.max)  # log scale
    logmel = librosa.util.fix_length(logmel, size=173, axis=1)  # pad/truncate
    return logmel.T   # (time_steps, n_mels)

# === Loop over synthetic dataset ===
for file in os.listdir(synthetic_dir):
    if file.endswith(".wav"):
        audio_path = os.path.join(synthetic_dir, file)
        csv_path = audio_path.replace(".wav", "_timestamps.csv")

        # Load audio + metadata
        y_audio, _ = librosa.load(audio_path, sr=SR)
        timestamps = pd.read_csv(csv_path)

        total_duration = librosa.get_duration(y=y_audio, sr=SR)
        step = int(WINDOW_SIZE * SR)
        hop = int(HOP_SIZE * SR)

        for start in range(0, len(y_audio) - step, hop):
            end = start + step
            segment = y_audio[start:end]

            # Segment timing (in seconds)
            seg_start = start / SR
            seg_end = end / SR

            # Match label from timestamp CSV
            label_row = timestamps[
                (timestamps["start_time"] <= seg_start) &
                (timestamps["end_time"] >= seg_end)
            ]

            if len(label_row) > 0:
                label = label_row.iloc[0]["label"]

                # Extract log-mel features
                logmel_features = extract_logmel(segment, SR)
                X.append(logmel_features)
                y.append(label)

# === Convert to arrays ===
X = np.array(X)
y = np.array(y)

# === Save to disk ===
np.save(save_X_path, X)
np.save(save_y_path, y)

print("✅ Preprocessing complete & saved")
print("X shape:", X.shape)   # (num_segments, time_steps, n_mels)
print("y shape:", y.shape)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# === Load preprocessed features ===
X = np.load("/content/drive/MyDrive/CRNN_X.npy")
y = np.load("/content/drive/MyDrive/CRNN_y.npy")

print("Before encoding:", X.shape, y.shape)

# === Encode labels to integers ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=len(le.classes_))

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# === Reshape for CRNN (need channel dimension) ===
X_train = X_train[..., np.newaxis]   # (samples, time_steps, n_mels, 1)
X_test  = X_test[..., np.newaxis]

print("After processing:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)
print("Classes:", le.classes_)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.layers import Reshape, Bidirectional, LSTM, Dense, Input

model = Sequential()

# === CNN Layers ===
model.add(Input(shape=(173, 64, 1)))

model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

# === Reshape dynamically ===
# Flatten frequency axis, keep reduced time axis
model.add(Reshape((-1, 128)))   # (time_steps, features) → Keras infers time dimension automatically

# === BiLSTM Layers ===
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dropout(0.3))

# === Dense Layers ===
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(len(le.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# === Callbacks ===
checkpoint_path = "/content/drive/MyDrive/best_crnn_model.h5"
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath=checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1)
]

# === Train model ===
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# === Plot training history ===
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend(); plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend(); plt.title("Loss")

plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

import os
import json
import joblib
import pickle

# === Create Save Directory ===
save_dir = "/content/drive/MyDrive/CRNN_saved"
os.makedirs(save_dir, exist_ok=True)

# === 1. Save Trained CRNN Model ===
model.save(os.path.join(save_dir, "best_crnn_model.h5"))

# === 2. Save Training History ===
with open(os.path.join(save_dir, "crnn_training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

# === 3. Save Label Encoder ===
joblib.dump(le, os.path.join(save_dir, "label_encoder.pkl"))

# === 4. Save Feature Extraction Parameters ===
feature_params = {
    "feature_type": "log_mel_spectrogram",
    "n_mels": 64,
    "hop_length": 512,
    "n_fft": 2048,
    "duration": 1.0,
    "sr": 22050,
    "window_size": 1.0,
    "hop_size": 0.5
}
with open(os.path.join(save_dir, "feature_params.json"), "w") as f:
    json.dump(feature_params, f)

# === 5. Save Class Mapping ===
class_mapping = {i: cls for i, cls in enumerate(le.classes_)}
with open(os.path.join(save_dir, "class_mapping.json"), "w") as f:
    json.dump(class_mapping, f)

print("✅ All CRNN components saved successfully to:", save_dir)

import os
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.models import load_model

# === Paths ===
synthetic_dir = "/content/drive/MyDrive/part2dumb/CRNN_dataset"
model_path = "/content/drive/MyDrive/part2dumb/CRNN_saved/best_crnn_model.h5"
SR = 22050
WINDOW_SIZE = 1.0   # seconds (must match training)
HOP_SIZE = 0.5      # seconds (must match training)

# === Load model ===
model = load_model(model_path)

# === Load LabelEncoder classes ===
# Replace with your saved label encoder if available
classes = ['air_conditioner','car_horn','children_playing','dog_bark','drilling',
           'engine_idling','gun_shot','jackhammer','siren','street_music']

# === Function to extract log-mel spectrogram (same as training) ===
import librosa
def extract_logmel(y_segment, sr, n_mels=64):
    mel = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = librosa.util.fix_length(logmel, size=173, axis=1)
    return logmel.T

# === Loop over synthetic audios ===
for file in os.listdir(synthetic_dir):
    if file.endswith(".wav"):
        audio_path = os.path.join(synthetic_dir, file)
        y_audio, _ = librosa.load(audio_path, sr=SR)
        total_duration = librosa.get_duration(y=y_audio, sr=SR)

        step = int(WINDOW_SIZE * SR)
        hop = int(HOP_SIZE * SR)

        timestamps = []

        # Sliding window over audio
        for start in range(0, len(y_audio) - step, hop):
            end = start + step
            segment = y_audio[start:end]

            # Extract features
            features = extract_logmel(segment, SR)
            features = features[np.newaxis, ..., np.newaxis]  # (1, time_steps, n_mels, 1)

            # Predict
            pred_prob = model.predict(features, verbose=0)
            pred_label_idx = np.argmax(pred_prob, axis=1)[0]
            pred_label = classes[pred_label_idx]

            seg_start = start / SR
            seg_end = end / SR

            timestamps.append({
                "start_time": round(seg_start, 2),
                "end_time": round(seg_end, 2),
                "predicted_label": pred_label
            })

        # Save predictions to CSV
        pred_df = pd.DataFrame(timestamps)
        csv_out_path = os.path.join(synthetic_dir, file.replace(".wav","_predictions.csv"))
        pred_df.to_csv(csv_out_path, index=False)
        print(f"✅ Predictions saved: {csv_out_path}")

"""Anotehr"""

# Cell A: Train CRNN with 2x BiLSTM + Attention + SpecAugment
# Paste and run this cell as-is.

import os
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import joblib
import pickle

# ========== 1) Load dataset or use in-memory variables ==========
if all(var in globals() for var in ("X_train","X_test","y_train","y_test")):
    print("Using in-memory X_train/X_test/y_train/y_test")
    # ensure shapes and dtypes
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    # try to get label encoder if exists
    if 'le' in globals():
        le_local = le
    else:
        # try to recover class names from y_test shape (not possible) -> fallback
        le_local = None
else:
    print("Loading CRNN_X.npy and CRNN_y.npy from Drive")
    X = np.load("/content/drive/MyDrive/CRNN_X.npy", allow_pickle=False)
    y = np.load("/content/drive/MyDrive/CRNN_y.npy", allow_pickle=False)
    # encode labels
    le_local = LabelEncoder()
    y_enc = le_local.fit_transform(y)
    y_cat = to_categorical(y_enc, num_classes=len(le_local.classes_))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_enc
    )
    # add channel dim
    X_train = X_train[..., np.newaxis].astype(np.float32)
    X_test  = X_test[..., np.newaxis].astype(np.float32)
    le = le_local  # store globally for later use
    print("Loaded and prepared data. Classes:", le_local.classes_)

print("Shapes -> X_train:", X_train.shape, "X_test:", X_test.shape, "y_train:", y_train.shape, "y_test:", y_test.shape)

# ========== 2) SpecAugment helper ==========
def spec_augment(logmel, time_mask_param=20, freq_mask_param=8, n_time_masks=1, n_freq_masks=1):
    # logmel: (T, F)
    X = logmel.copy()
    T, F = X.shape
    for _ in range(n_time_masks):
        t = random.randint(0, min(time_mask_param, max(0, T-1)))
        if t == 0: continue
        t0 = random.randint(0, max(0, T - t))
        X[t0:t0+t, :] = 0.0
    for _ in range(n_freq_masks):
        f = random.randint(0, min(freq_mask_param, max(0, F-1)))
        if f == 0: continue
        f0 = random.randint(0, max(0, F - f))
        X[:, f0:f0+f] = 0.0
    return X

# ========== 3) Attention layer ==========
from tensorflow.keras.layers import Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        # input_shape: (batch, time, features)
        self.w = self.add_weight(shape=(input_shape[-1],), initializer='random_normal', trainable=True, name='att_w')
        super().build(input_shape)
    def call(self, inputs):
        # inputs: (batch, time, features)
        score = tf.tensordot(inputs, self.w, axes=1)        # (batch, time)
        weights = tf.nn.softmax(score, axis=1)              # (batch, time)
        weighted = inputs * tf.expand_dims(weights, -1)     # (batch, time, features)
        return tf.reduce_sum(weighted, axis=1)              # (batch, features)

# ========== 4) DataGenerator with SpecAugment ==========
from tensorflow.keras.utils import Sequence
import math

class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size=16, shuffle=True, augment=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.X))
        self.on_epoch_end()
    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)
    def __getitem__(self, idx):
        batch_idx = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_X = np.zeros((len(batch_idx),) + self.X.shape[1:], dtype=np.float32)
        for i, ii in enumerate(batch_idx):
            sample = self.X[ii]
            s = sample.squeeze(-1)  # (T,F)
            if self.augment:
                s = spec_augment(s, time_mask_param=20, freq_mask_param=8)
            batch_X[i] = s[..., np.newaxis]
        batch_y = self.y[batch_idx]
        return batch_X, batch_y
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# set shuffle=False for validation generator to preserve order
train_gen = DataGenerator(X_train, y_train, batch_size=16, shuffle=True, augment=True)
val_gen   = DataGenerator(X_test,  y_test,  batch_size=16, shuffle=False, augment=False)

# ========== 5) Build model_att (2x BiLSTM + Attention) ==========
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Reshape, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model

def build_crnn_att(input_shape, n_classes, dropout=0.3):
    x_in = Input(shape=input_shape)   # e.g., (173, 64, 1)

    # Block 1
    x = Conv2D(32, (3,3), padding='same', activation='relu')(x_in)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(dropout)(x)

    # Block 2
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(dropout)(x)

    # Block 3
    x = Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(dropout)(x)

    # Reshape: (batch, time', freq', ch) -> (batch, time', features)
    shape = K.int_shape(x)   # (None, time', freq', ch)
    time_steps = shape[1]
    features = shape[2] * shape[3]
    x = Reshape((time_steps, features))(x)

    # 2 × BiLSTM
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(dropout)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(dropout)(x)

    # Attention
    x = AttentionLayer()(x)

    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout)(x)
    out = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=x_in, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


input_shape = X_train.shape[1:]  # (T, F, 1)
n_classes = y_train.shape[1]
model = build_crnn_att(input_shape=input_shape, n_classes=n_classes, dropout=0.3)
model.summary()

# ========== 6) Callbacks and training ==========
save_dir = "/content/drive/MyDrive/CRNN_second_saved"
os.makedirs(save_dir, exist_ok=True)
checkpoint_path = os.path.join(save_dir, "crnn_att_best.h5")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=80,
    callbacks=callbacks,
    verbose=1
)

# ========== 7) Save artifacts ==========
model.save(os.path.join(save_dir, "crnn_att_final.h5"))
with open(os.path.join(save_dir, "crnn_att_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)
if 'le' in globals():
    joblib.dump(le, os.path.join(save_dir, "label_encoder.pkl"))

# ========== 8) Evaluate and show results ==========
test_loss, test_acc = model.evaluate(val_gen, verbose=1)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%  |  Test Loss: {test_loss:.4f}")

# predictions
y_pred_probs = model.predict(val_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification report:")
if 'le' in globals():
    labels = le.classes_
else:
    labels = [str(i) for i in range(n_classes)]
print(classification_report(y_true, y_pred, target_names=labels))

# confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.show()

"""Efficinetnetb0"""

import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Update this path to where UrbanSound8K is mounted in Colab/Local
dataset_path = "/content/drive/MyDrive/archive"
metadata = pd.read_csv(os.path.join(dataset_path, "UrbanSound8K.csv"))

print(metadata.head())

SAMPLE_RATE = 22050
DURATION = 4  # seconds
SAMPLES = SAMPLE_RATE * DURATION
N_MELS = 128

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Pad or trim to fixed length
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]

    # Convert to mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize 0–1
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())

    # Convert grayscale → 3 channel
    return np.stack([log_mel, log_mel, log_mel], axis=-1)

X, y = [], []

for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
    file_path = os.path.join(dataset_path, f"fold{row['fold']}", row['slice_file_name'])

    try:
        spec = preprocess_audio(file_path)
        X.append(spec)
        y.append(row["class"])
    except Exception as e:
        print("Error:", file_path, e)

X = np.array(X)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Save for reuse
np.save(os.path.join(SAVE_DIR, "X.npy"), X)
np.save(os.path.join(SAVE_DIR, "y.npy"), y_cat)
joblib.dump(encoder, os.path.join(SAVE_DIR, "label_encoder.pkl"))

print("✅ Dataset ready:", X.shape, y_cat.shape)

import os, joblib
import numpy as np

# Define SAVE_DIR again
SAVE_DIR = "/content/drive/MyDrive/efficientnet_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# Save dataset + label encoder
np.save(os.path.join(SAVE_DIR, "X.npy"), X)
np.save(os.path.join(SAVE_DIR, "y.npy"), y_cat)
joblib.dump(encoder, os.path.join(SAVE_DIR, "label_encoder.pkl"))

print("✅ Saved processed dataset in:", SAVE_DIR)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y_cat, random_state=42
)

print("Train:", X_train.shape, "Test:", X_test.shape)

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

# Base EfficientNetB0
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 173, 3)  # (n_mels, time, channels)
)

# Freeze base initially
base_model.trainable = False

# Add custom classifier
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(y_cat.shape[1], activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

"""selvd_Saved"""

!pip install librosa soundfile

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import json
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

print("✅ Setup complete!")
print(f"TensorFlow version: {tf.__version__}")

METADATA_PATH = "/content/drive/MyDrive/archive/UrbanSound8K.csv"
AUDIO_DIR = "/content/drive/MyDrive/archive"
SAVE_DIR = "/content/drive/MyDrive/SELD_saved"

os.makedirs(SAVE_DIR, exist_ok=True)

metadata = pd.read_csv(METADATA_PATH)
print(f"✅ Dataset loaded: {len(metadata)} samples")
print(f"Classes: {metadata['class'].unique()}")
print(f"Folds: {metadata['fold'].unique()}")

sample_file = os.path.join(AUDIO_DIR, f"fold{metadata.iloc[0]['fold']}", metadata.iloc[0]['slice_file_name'])
print(f"Sample file exists: {os.path.exists(sample_file)}")

def extract_seld_features(metadata, audio_dir, sr=22050, window_length=2.0, hop_length=0.5):
    """
    Extract features optimized for SELD task
    Creates sliding windows with multi-label detection + localization labels
    """

    # Feature extraction parameters (optimized for 90%+ accuracy)
    n_fft = 2048
    hop_len = 512  # ~23ms
    n_mels = 128   # Higher resolution

    X_windows = []
    y_detection = []  # Multi-hot labels per window
    y_localization = []  # Start/end frame labels

    print("Extracting SELD features...")

    for idx, row in metadata.iterrows():
        if idx % 1000 == 0:
            print(f"Processing {idx}/{len(metadata)} files...")

        # Load audio file
        audio_path = os.path.join(audio_dir, f"fold{row['fold']}", row['slice_file_name'])

        try:
            y, _ = librosa.load(audio_path, sr=sr, duration=4.0)  # Load max 4s
        except:
            print(f"Skipped: {audio_path}")
            continue

        # Pad or truncate to fixed length for windowing
        target_length = int(4.0 * sr)  # 4 seconds
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]

        # Create sliding windows
        window_samples = int(window_length * sr)
        hop_samples = int(hop_length * sr)

        for start_sample in range(0, len(y) - window_samples + 1, hop_samples):
            window_audio = y[start_sample:start_sample + window_samples]

            # Extract log-mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=window_audio, sr=sr, n_fft=n_fft,
                hop_length=hop_len, n_mels=n_mels
            )
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)

            # Normalize
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

            X_windows.append(log_mel.T)  # Shape: (time_frames, n_mels)

            # Create labels for this window
            window_start_time = start_sample / sr
            window_end_time = (start_sample + window_samples) / sr

            # Detection label (multi-hot for overlapping events)
            detection_label = np.zeros(10)
            detection_label[row['classID']] = 1  # This class is present
            y_detection.append(detection_label)

            # Localization label (simplified: event spans entire original clip)
            # In real SELD, you'd have precise start/end times
            localization_label = np.zeros(10 * 2)  # start/end for each class

            # Mark event boundaries (simplified for UrbanSound8K)
            class_id = row['classID']
            # Event starts at beginning of window
            localization_label[class_id * 2] = 1.0      # Start
            localization_label[class_id * 2 + 1] = 1.0  # End

            y_localization.append(localization_label)

    return np.array(X_windows), np.array(y_detection), np.array(y_localization)

# Extract features
print("Starting feature extraction...")
X, y_detection, y_localization = extract_seld_features(metadata, AUDIO_DIR)

print(f"✅ Feature extraction complete!")
print(f"X shape: {X.shape}")
print(f"Detection labels shape: {y_detection.shape}")
print(f"Localization labels shape: {y_localization.shape}")

# Save features
np.save(os.path.join(SAVE_DIR, "SELD_X_features.npy"), X)
np.save(os.path.join(SAVE_DIR, "SELD_y_detection.npy"), y_detection)
np.save(os.path.join(SAVE_DIR, "SELD_y_localization.npy"), y_localization)

print("✅ Features saved to Drive!")

"""part 2

"""

import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

# Paths (modify if needed)
metadata_path = "/content/drive/MyDrive/archive/UrbanSound8K.csv"
audio_dir = "/content/drive/MyDrive/archive"
save_segments_path = "/content/drive/MyDrive/segments.csv"

# Parameters from your feature_params.json and Part 1
sr = 16000
n_mfcc = 40
hop_length = 512
n_fft = 2048
window_duration = 4.0  # seconds
# Calculate duration of mfcc frame
mfcc_frame_duration = hop_length / sr  # ~0.032s
n_mfcc_frames = 173  # from your MFCC fixed length

# Assuming 50% overlap in window extraction (adjust if needed)
hop_duration = window_duration / 2  # 2 seconds if 50% overlap

# Augmentation function (copy from your Part 1)
def augment_audio(y, sr):
    methods = []

    # Pitch shift
    n_steps = random.choice([-2, -1, 1, 2])
    shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    methods.append(shifted)

    # Time stretch
    rate = random.uniform(0.9, 1.1)
    try:
        stretched = librosa.effects.time_stretch(y, rate)
        methods.append(stretched)
    except:
        pass

    # Add noise
    noise = np.random.normal(0, 0.005, y.shape)
    methods.append(y + noise)

    return methods

# Feature extraction with augmentation (copy from Part 1)
def extract_features(file_path, augment=False):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        features = []

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = librosa.util.fix_length(mfcc, size=n_mfcc_frames, axis=1)
        features.append(mfcc)

        if augment:
            augmented_versions = augment_audio(y, sr)
            for aug_y in augmented_versions:
                mfcc_aug = librosa.feature.mfcc(y=aug_y, sr=sr, n_mfcc=n_mfcc)
                mfcc_aug = librosa.util.fix_length(mfcc_aug, size=n_mfcc_frames, axis=1)
                features.append(mfcc_aug)

        return features, y, sr
    except Exception as e:
        print(f"⚠️ Error in {file_path}: {e}")
        return [], None, None

# Converts a list of window class labels to timestamped segments
def windows_to_segments(predicted_classes, window_duration, hop_duration):
    segments = []
    if len(predicted_classes) == 0:
        return pd.DataFrame(columns=['start_time', 'end_time', 'class'])

    current_class = predicted_classes[0]
    start_time = 0.0

    for i in range(1, len(predicted_classes)):
        if predicted_classes[i] != current_class:
            end_time = start_time + window_duration + (i-1) * hop_duration - start_time
            segments.append({
                'start_time': start_time,
                'end_time': i * hop_duration + window_duration,
                'class': current_class
            })
            current_class = predicted_classes[i]
            start_time = i * hop_duration

    # Last segment
    segments.append({
        'start_time': start_time,
        'end_time': start_time + window_duration,
        'class': current_class
    })

    return pd.DataFrame(segments)

# Load metadata
metadata = pd.read_csv(metadata_path)

all_segments = []

print("\n🔁 Extracting features and generating segments...")
for i, row in tqdm(metadata.iterrows(), total=len(metadata)):

    fold = f"fold{row['fold']}"
    file_name = row["slice_file_name"]
    true_label = row["classID"]

    file_path = os.path.join(audio_dir, fold, file_name)
    mfcc_list, y_audio, sr_audio = extract_features(file_path, augment=False)
    if not mfcc_list:
        continue

    # Simulate predicted classes as true class (for now)
    predicted_classes = [true_label] * len(mfcc_list)  # Replace with your actual model predictions if available

    # Convert to segments
    segments_df = windows_to_segments(predicted_classes, window_duration, hop_duration)
    segments_df['file_name'] = file_name
    all_segments.append(segments_df)

# Concatenate all and save
final_segments = pd.concat(all_segments, ignore_index=True)
final_segments.to_csv(save_segments_path, index=False)
print(f"\n✅ Saved segment timestamps and classes to: {save_segments_path}")

# Output example
print(final_segments.head())