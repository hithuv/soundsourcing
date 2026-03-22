# Separation Models Explained

This document explains the baseline and all benchmarked alternatives in this repository.

## 1) Baseline U-Net (`baseline_unet`)

- **Input**: mixture magnitude spectrogram `[1, F, T]`.
- **Core idea**: encoder-decoder with skip connections predicts `num_classes` masks.
- **Output**: class-wise masks `[C, F, T]`; each class magnitude is `mix_mag * mask`.
- **Strength**: simple and stable; fast enough for local M3 Pro iteration.
- **Weakness**: limited context modeling and skip features can pass interference.

## 2) Residual Attention U-Net (`resattn_unet`)

- **What changed vs baseline**:
  - residual blocks at each level for stronger feature reuse,
  - attention gates on skip connections to suppress irrelevant skip activations.
- **Why it can improve**:
  - deeper representational power without severe optimization instability,
  - cleaner skip fusion helps with overlapping sources.
- **Compute profile**: moderately heavier than baseline.

## 3) TFC-TDF U-Net (`tfctdf_unet`)

- **Core idea**: combines standard time-frequency convolutions (TFC) with directional frequency/time transforms (TDF-like blocks).
- **Why it can improve**:
  - captures joint local texture and directional structure in spectrograms,
  - better handling of transient and harmonic regions.
- **Compute profile**: moderate; typically slower than baseline but manageable on M3 Pro.

## 4) Conv-TasNet Lite (`conv_tasnet`)

- **Domain**: waveform (time domain), not spectrogram masks.
- **Core idea**:
  - encode waveform into latent frames,
  - temporal convolution stack predicts class-wise masks in latent space,
  - decode to class-wise waveforms.
- **Why it can improve**:
  - does not depend on mixture phase reuse,
  - often better reconstruction of fine temporal details.
- **Compute profile**: heavier than spectrogram models.

## 5) Demucs-Lite (`demucs_lite`)

- **Domain**: waveform with encoder-decoder hierarchy.
- **Core idea**: hierarchical downsample/upsample temporal features with skip connections and residual conv blocks.
- **Why it can improve**:
  - large temporal receptive field,
  - strong detail recovery through decoder and skips.
- **Compute profile**: heavy among tested options; use balanced preset for practical runs.

## How to interpret benchmark metrics

- **SI-SDR**: scale-invariant source quality; robust primary ranking metric.
- **SDR**: source-to-distortion ratio.
- **SIR**: source-to-interference ratio.

Higher is better for all three. Use SI-SDR as the main selection signal and confirm with listening tests from exported audio examples.

