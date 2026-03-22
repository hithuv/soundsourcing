
Below is a **paper-style monograph** you can adapt into a ~10-page PDF: it ties together the **task**, **data**, **models**, **training**, **metrics**, **results**, and **where each artifact lives** in your repo. Numbers come from your exported benchmark (`results/paper/table_benchmark.csv`) and the evaluation pipeline described in code.

---

# Environmental Sound Separation with UrbanSound8K–Derived Mixtures: Models, Training, and Results

## 1. Introduction and scope

This project studies **source separation** in an **urban sound** setting derived from **UrbanSound8K (US8K)**. Unlike classical speech separation (two speakers, fixed sampling), the task is **multi-class**: the model must explain the mixture as **per-class energy** (10 classes) so that each class channel approximates the isolated source for that class when it is present.

Work spans: **dataset construction**, **five separation architectures** (three spectrogram maskers, two time-domain models), **unified training** (`experiments/train/train_utils.py`), **validation metrics** (training loss plus **waveform SI-SDR** on a held-out split), and **paper-oriented exports** (`results/paper/`).

**Primary code locations**

| Area | Path |
|------|------|
| Experiment config | `experiments/configs/defaults.py` |
| Dataset & splits | `experiments/data/separation_dataset.py` |
| Training loop & loss | `experiments/train/train_utils.py` |
| Models | `experiments/models/*.py` |
| SI-SDR benchmark | `experiments/eval/compare_separation_models.py` |
| Paper tables/figures | `results/paper/` |
| Mixture dataset script | `1b_create_separation_dataset.py` |

---

## 2. Problem formulation

### 2.1 What the network predicts

- **Input:** A mixture waveform \(x(t)\) (or its magnitude spectrogram for spec models).
- **Output:** **10 channels**—one estimate per US8K class. For spectrogram models, outputs are **masks** \(M_c \in [0,1]^{F\times T}\) applied to the mixture magnitude; for waveform models, **10 waveforms** \(\hat{s}_c(t)\).

### 2.2 Supervision

For each mixture, **ground-truth per-class waveforms** are built by loading each contributing slice, assigning it to its **class label**, and summing within that class (`SeparationDataset`). **Inactive classes** are (near) silent. Training therefore encourages each channel to match **only** the energy that belongs to that class.

This is a form of **multi-output separation** with a **fixed class dictionary** (the 10 US8K classes), not a permutation-invariant two-source setup like many speech papers.

---

## 3. Dataset: UrbanSound8K separation mixtures

### 3.1 Construction (`1b_create_separation_dataset.py`)

- **Source metadata:** `UrbanSound8K/metadata/UrbanSound8K.csv`
- **Output:** `UrbanSound8K_separation/` with `audio/` and `metadata.csv`
- **Protocol:** For each synthetic example, sample **2 or 3** sources such that **each of the chosen classes appears once** (`groupby('classID').sample(1)`), normalize each clip, sum, renormalize mixture.
- **Scale:** `NUM_SYNTHETIC_FILES = 3000`, **22.05 kHz**, **4 s** clips (`SR = 22050`, `DURATION = 4`).

**Novelty angle (for your report):** You are not using official US8K “classification only” splits as separation; you **synthesize controlled multi-source mixtures** with **known source paths**, enabling **supervised separation** with **class-aligned targets**.

### 3.2 Training/validation split (`separation_dataset.py`, `defaults.py`)

- Metadata is split: **train / val / test** via `train_test_split` with **15% val, 15% test**, `random_seed=42`.
- Each item provides: `mix_wave`, `source_waves` `[10, T]`, `mix_mag`, `target_mags` for STFT-based losses.

**File references:** `UrbanSound8K_separation/metadata.csv` (if committed), `experiments/data/separation_dataset.py`.

---

## 4. Experimental setup (what was held constant)

From `BALANCED_PRESET` (`experiments/configs/defaults.py`):

| Setting | Value |
|--------|--------|
| Sample rate | 22050 Hz |
| Clip length | 4 s → 88200 samples |
| STFT | `n_fft=1024`, `hop_length=256` |
| Classes | 10 |
| Batch size | 4 |
| Epochs | 30 (max) |
| Optimizer | AdamW, lr `1e-3`, weight decay `1e-5` |
| Scheduler | Cosine annealing over epochs |
| Early stopping | patience 6 (on validation loss) |
| Device | Apple **MPS** when available (`select_device`) |

**Runs** are stored under `results/<timestamp>_<model>_balanced/` with `config.json`, `metrics/train_history.csv`, `metrics/train_summary.json`, and `checkpoints/best.pth` (checkpoints often git-ignored).

---

## 5. Architectures: baseline and extensions

### 5.1 Baseline U-Net (spectrogram masking)

**File:** `experiments/models/baseline_unet.py`

- **Idea:** Standard **U-Net** on **single-channel magnitude** input: encoder–decoder with skip connections, **sigmoid** on the output so each of **10 classes** produces a **mask** in \([0,1]\).
- **Role:** Simple, strong **baseline** for spectrogram-conditioned separation; widely used in audio ML.

**Why it might work:** Encoder–decoder structure captures **local patterns** in time–frequency; skip links preserve **spatial detail** for mask estimation.

**Limitation:** No explicit **long-range** or **frequency-vs-time** factorization; a single receptive field at each layer.

---

### 5.2 Residual Attention U-Net (`resattn_unet`)

**File:** `experiments/models/resattn_unet.py`

- **Ideas combined:**
  - **Residual blocks** (`ResidualBlock`) for easier optimization and deeper effective representation.
  - **Attention gates** (`AttentionGate`) on skip connections—**inspired by attention U-Nets** in medical imaging / audio (gate emphasizes informative skip regions).

**Details:** At each decoding stage, the skip tensor is **scaled by a learned attention map** derived from gate and skip (`skip * alpha`).

**Why it *could* beat baseline:** Attention can **suppress irrelevant** time–frequency regions in skips and focus on **class-salient** structure.

**What you observed in practice:** Training **validation loss** for ResAttn and TFCTDF landed **almost identical** to baseline (~0.1011); SI-SDR also **ties** baseline-level models (see §7). So in *this* setup, the extra machinery did **not** translate into a clear gain—useful **negative result** for the paper (architecture vs data/task mismatch).

---

### 5.3 TF–CTDF U-Net (`tfctdf_unet`)

**File:** `experiments/models/tfctdf_unet.py`

- **TFC–TDF block:** Alternates **time–frequency conv** (`TFC`) with **separable time/frequency** convs (`Tdf`: e.g. `(1,5)` then `(5,1)`), plus **residual** wrapping.
- **Inspiration:** Lines of work on **time–frequency processing** that treat **frequency structure** and **temporal structure** with different kernels (related to **TFC–TDF** ideas in music source separation literature, in a lightweight form here).

**Why it *could* beat baseline:** Explicit **anisotropic** filtering along **time vs frequency** can match **harmonic vs transient** structure better than isotropic 3×3 stacks alone.

**Your runs:** Again **~same best val loss** as baseline; **early stopping** at epoch 7 with **best epoch 1** for both ResAttn and TFCTDF (per `table_benchmark.csv`), suggesting **quick plateau** and **no sustained improvement** from extra complexity.

---

### 5.4 Conv-TasNet (lite) — time-domain

**File:** `experiments/models/conv_tasnet.py`

- **Inspiration:** **Conv-TasNet**-style **time-domain** separation: encoder (1-D conv) → **temporal convolutional** stack with **depthwise separable** dilated convs → **mask** in encoder space → decoder (transpose conv) to waveforms.
- **Output:** `[B, 10, T]` waveforms (after alignment in loss).

**Training loss:** **Waveform L1** vs `source_waves` (`compute_loss` in `train_utils.py`)—**not** comparable in magnitude to spectrogram losses (see §6).

**Why time-domain can help:** No fixed STFT phase; model can learn **analysis filters**. Often strong on **speech**; for **environmental** sounds, data and capacity matter.

**Your SI-SDR:** **Worse** than the best spectrogram models on your SI-SDR eval (~**−4.46 dB** mean), despite **similar waveform L1** to others (~0.0493). This is a key **discussion point**: **L1 ≠ perceptual / SI-SDR alignment**.

---

### 5.5 Demucs (lite) — time-domain encoder–decoder

**File:** `experiments/models/demucs_lite.py`

- **Inspiration:** **Demucs**-family **encoder–decoder over waveforms** with **strided conv blocks** and skip fusion—here a **small** variant (not full hybrid Demucs).

**Bug fix you applied:** `ConvBlock1d` used **k=8, stride=1** with padding that **shortened** length by 1, breaking residual `net(x) + x`. **Fix:** use **k=7, padding=3** when `stride==1` so residual paths stay **length-aligned**.

**Training loss:** Waveform L1; **best val loss** ~**0.0123**, comparable to Conv-TasNet.

**SI-SDR:** **Very poor** (~**−26.74 dB**) while **waveform L1** is still ~0.0493. That pattern usually indicates **wrong scaling / phase / interference** under SI-SDR even when **absolute error** looks similar—excellent **analysis subsection** for “why metrics disagree.”

---

## 6. Training objectives: why numbers are not all comparable

**Spec models** (`SPEC_MODELS` in `train_utils.py`): `baseline_unet`, `resattn_unet`, `tfctdf_unet`.

\[
\mathcal{L} = 0.7 \cdot \text{L1}(|\hat{X}|, |S|) + 0.3 \cdot \text{L1}(M, \text{clip}(|S|/|X|, 0, 1))
\]

where \(\hat{X} = M \odot |X|\) (mixture magnitude times mask). This blends **reconstruction in magnitude** with **mask-ratio** supervision.

**Wave models** (`conv_tasnet`, `demucs_lite`): **mean absolute error** between predicted and target **waveforms** per channel.

**Implication:** A val loss of **~0.10** (spec) vs **~0.012** (wave) does **not** mean wave models are “10× better”—they optimize **different targets**.

**Files:** `experiments/train/train_utils.py` (`compute_loss`).

---

## 7. Evaluation: SI-SDR and what you reported

**Script:** `experiments/eval/compare_separation_models.py`

- **Metric:** **SI-SDR** (scale-invariant SDR) in **dB**, averaged over **(example, class)** pairs where the **reference RMS > threshold** (default `1e-3`), so silent classes do not dominate.
- **Spec models:** Reconstruct waveform via **ISTFT** of \((\texttt{mix\_mag} \odot M)\) with **mixture phase** (implementation matches training reconstruction intent).
- **Outputs:** `results/separation_model_comparison.json`, merged into `results/paper/table_benchmark.csv`.

**Paper assets:** `results/paper/fig_training_curves.{pdf,png}`, `fig_si_sdr_validation.{pdf,png}`, `table_benchmark.tex`, `benchmark_summary.json` — generator: `experiments/paper/export_paper_assets.py`.

---

## 8. Quantitative results (your benchmark)

From `results/paper/table_benchmark.csv` (paths on disk are under `results/...` in your run):

| Model | Best val loss (training) | Best epoch | Epochs run | Mean SI-SDR (dB) | Mean WF L1 (eval) |
|--------|---------------------------|------------|------------|------------------|-------------------|
| Baseline U-Net | 0.1011 | 30 | 30 | **−0.21** | 0.04930 |
| ResAttn U-Net | 0.1011 | 1 | 7 | **~0** | 0.04930 |
| TF–CTDF U-Net | 0.1011 | 1 | 7 | **~0** | 0.04930 |
| Conv-TasNet (lite) | 0.0123 | 11 | 17 | **−4.46** | 0.04929 |
| Demucs (lite) | 0.0123 | 30 | 30 | **−26.74** | 0.04930 |

**SI-SDR pairs counted:** 1123 (same for all models in that export).

### 8.1 How to narrate this

1. **Spectrogram U-Nets (three variants)** cluster at **~0 dB** SI-SDR (weak absolute separation) with **nearly identical** training loss—**sophisticated blocks did not beat the simple U-Net** on these metrics. Honest **ablation story:** inductive bias did not pay off under **fixed data, capacity, and objective**.

2. **Conv-TasNet (lite)** has **lower training loss** in its own objective but **worse SI-SDR** than the best spec models—shows **objective mismatch** and possibly **under-training / capacity** for time-domain on this task.

3. **Demucs (lite)** is **catastrophic on SI-SDR** despite **similar L1** to others—strong evidence that **waveform L1 alone is insufficient** and that the **lite** architecture or **optimization** may produce **biased** waveforms that still have low L1.

---

## 9. Why things might work or fail (discussion for “paper quality”)

### 9.1 Plateau and early stopping

Several models hit **best validation at epoch 1** and then **early stopped** (patience 6). That suggests **limited room** in the current objective, **learning rate schedule** (cosine from the start), or **need for stronger augmentation / longer training / different loss** (e.g. SI-SDR loss, multi-resolution STFT).

### 9.2 Multi-class vs two-source separation

Your mixture has **2–3 sources** but **10 outputs**. Inactive channels are silent; the model may still allocate **energy** incorrectly across classes—**permutation / interference** between classes is not handled by a **PIT** loss (permutation invariant training). **Novelty / future work:** PIT across active classes, or **two-head** models for unknown \(K\).

### 9.3 Phase

Spec models use **mixture phase** for ISTFT—**classic limitation**; wrong phase limits quality. Time-domain models avoid fixed STFT phase but need **capacity and data**.

### 9.4 “Novelty” you can claim (appropriately scoped)

- **Synthetic US8K-based mixture dataset** with **class-labeled** targets for **multi-class** separation.
- **Systematic benchmark** of **U-Net / attention / TFC–TDF / Conv-TasNet lite / Demucs lite** under **one training framework** and **one SI-SDR evaluation**.
- **Critical finding:** **Architectural upgrades** did not improve validation loss or SI-SDR vs baseline; **metric analysis** shows **L1 and SI-SDR can disagree sharply** (Demucs case).

Frame novelty as **rigorous comparison + negative results + evaluation methodology**, not as state-of-the-art beating LibriMix.

---

## 10. Limitations (essential for academic tone)

- **Dataset size / diversity:** 3000 mixtures; generalization to **real recordings** is unclear.
- **Lite models:** Not full TasNet/Demucs/SepFormer scale.
- **Single condition:** One STFT, one lr schedule, one split seed.
- **SI-SDR** is one metric; **listening tests** or **SDR** variants would strengthen claims.
- **Checkpoints** may be omitted from git—reproducibility requires **training recipe** + **seed** + **config** (you have `config.json` per run).

---

## 11. File and artifact index (for your appendix)

| Content | Location |
|--------|--------|
| Mixture generation | `1b_create_separation_dataset.py` |
| Separation dataset & STFT | `experiments/data/separation_dataset.py` |
| Hyperparameters | `experiments/configs/defaults.py` |
| Training + loss | `experiments/train/train_utils.py` |
| Baseline U-Net | `experiments/models/baseline_unet.py` |
| ResAttn U-Net | `experiments/models/resattn_unet.py` |
| TFCTDF U-Net | `experiments/models/tfctdf_unet.py` |
| Conv-TasNet lite | `experiments/models/conv_tasnet.py` |
| Demucs lite | `experiments/models/demucs_lite.py` |
| SI-SDR evaluation | `experiments/eval/compare_separation_models.py` |
| Aggregated comparison JSON | `results/separation_model_comparison.json` |
| Paper table + figures | `results/paper/table_benchmark.csv`, `table_benchmark.tex`, `fig_*.pdf` |
| Per-run curves & summary | `results/<run>/metrics/train_history.csv`, `train_summary.json` |
| Benchmark orchestration | `experiments/run_benchmark.py` (if used) |

---

## 12. Suggested section outline for your 10-page PDF

1. **Abstract** – task, data, methods, main finding (baseline competitive; metrics disagree across objectives).
2. **Introduction** – environmental sound separation motivation.
3. **Related work** – U-Net separation, TasNet, Demucs, attention (cite papers in your bibliography).
4. **Dataset** – US8K, synthesis procedure, splits.
5. **Methods** – five architectures, loss functions, training details.
6. **Evaluation** – SI-SDR definition, active-class masking, ISTFT for spec models.
7. **Results** – table + training curves + bar chart (`results/paper/`).
8. **Discussion** – plateau, PIT, phase, Demucs vs L1.
9. **Conclusion & future work** – larger models, SI-SDR loss, real data.
10. **References + Appendix** – config table, file list.

---

You can paste sections into LaTeX/Word and trim to page limit; swap in your **institution name**, **author**, and **real citations** (UrbanSound8K, U-Net, Conv-TasNet, Demucs, attention U-Net, etc.). If you want, a follow-up pass can turn **§5–8** into a **LaTeX skeleton** with `\cite{}` placeholders only (no new files unless you ask in Agent mode).