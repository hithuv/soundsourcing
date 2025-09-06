
# UrbanSound8K: From Classification to Separation

This project explores a series of audio machine learning tasks using the UrbanSound8K dataset. It serves as a practical guide, progressing from basic single-sound classification to multi-label sound event detection, and finally to an experimental implementation of audio source separation.

The models are built with **PyTorch** and are optimized to leverage Apple Silicon (MPS) for local training on a MacBook.

## 🚧 Project Status
* **Single-Label Classification**: ✅ Fully functional. The model successfully classifies single sounds from the dataset.
* **Multi-Label Sound Event Detection**: ✅ Fully functional. The model can identify multiple sounds present in a synthetically generated audio clip.
* **Source Separation**: ⚠️ **Experimental.** The U-Net model trains, but it currently **does not effectively separate** audio sources. The output is very similar to the input. This part of the project serves as a learning sandbox for advanced audio architectures and highlights the challenges of source separation.

***

## Features
* **Single Sound Identification**: A simple CNN to classify 4-second audio clips into one of 10 urban sound classes.
* **Multi-Source Identification**: A multi-label CNN that detects all present sounds in a clip containing 2-3 mixed sources.
* **Source Separation (Experimental)**: A U-Net architecture designed to generate time-frequency masks to isolate or remove a specific sound source from a mix.
* **Data Synthesis Pipeline**: Scripts to automatically generate complex audio datasets (mixed sounds, separation pairs) from the base UrbanSound8K dataset.


***

## Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Download the Dataset
This project requires the UrbanSound8K dataset.
* You can download it from [this page](https://urbansounddataset.weebly.com/urbansound8k.html).
* After downloading, unzip the file and place the entire `UrbanSound8K` folder into the root of this project directory.

Your folder structure should look like this after this step:
```
AKHIL_SOUND/
├── UrbanSound8K/
│   ├── audio/
│   └── metadata/
├── train_sound_model.py
└── ... (other project files)
```


### 2. Set Up the Environment
It's highly recommended to use a Python virtual environment.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it (on macOS/Linux)
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python libraries using pip:

```bash
pip install torch torchvision torchaudio pandas numpy librosa scikit-learn tqdm soundfile
```

## Usage and Workflow

The project is divided into three parts. Run the scripts for each part in the specified order.

### Part 1: Single-Label Sound Classification

This model identifies a single sound in an audio clip.

1. **Train the Model**: Run the training script. It will save the trained model and a label encoder to the `output/` directory.
   ```bash
   python train_sound_model.py
   ```

2. **Test the Model**: Open `predict_sound.py`, change the `AUDIO_TO_TEST_RELATIVE` variable to a file from the `UrbanSound8K/audio/` directory, and run the script to see a prediction.
   ```bash
   python predict_sound.py
   ```

### Part 2: Multi-Label Sound Event Detection

This model identifies multiple sounds within a single clip.

1. **Generate Mixed Data**: This script creates a new `UrbanSound8K_mixed/` directory with audio files containing 2-3 mixed sounds.
   ```bash
   python 1_create_mixed_dataset.py
   ```

2. **Train the Multi-Label Model**: This trains a model on the new mixed dataset.
   ```bash
   python 2_train_multi_label_model.py
   ```

3. **Test the Multi-Label Model**: Open `3_test_multi_label_model.py`, change the `AUDIO_TO_TEST` variable to a file from the `UrbanSound8K_mixed/audio/` directory, and run it to see all detected sounds.
   ```bash
   python 3_test_multi_label_model.py
   ```

### Part 3: Experimental Source Separation

This attempts to remove a specific sound from a mixed clip. **Note**: This part is experimental and does not produce high-quality separation.

1. **Generate Separation Data**: This script creates the `UrbanSound8K_separation/` directory, which contains mixed audio and metadata linking to the clean original sources.
   ```bash
   python 1b_create_separation_dataset.py
   ```


2. **Train the U-Net Model**: This will take a significant amount of time. It trains the U-Net architecture to generate separation masks.
   ```bash
   python train_separation_model_4.py
   ```

3. **Attempt Separation**: Open `5_separate_audio.py`. Edit the `MIXED_AUDIO_PATH` and `CLASS_TO_REMOVE` variables, then run the script. It will produce an `output_separated.wav` file.
   ```bash
   python 5_separate_audio.py
   ```

## Technologies Used

* **PyTorch**: Core deep learning framework
* **Librosa**: The primary library for audio analysis and feature extraction
* **NumPy & Pandas**: For data manipulation and management
* **Scikit-learn**: For utilities like the LabelEncoder
* **SoundFile**: For writing the final separated audio files