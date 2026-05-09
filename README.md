# CSE425 Project: Unsupervised Neural Network for Multi-Genre Music Generation

## 1. Project Overview

This project implements a symbolic music generation system using neural network models trained on the MAESTRO MIDI dataset. The system generates playable MIDI music using different generative modelling approaches and compares their performance using symbolic music evaluation metrics.

The project follows the supplementary implementation guide for **Unsupervised Neural Network for Multi-Genre Music Generation** and includes four major tasks:

1. **Task 1: LSTM Autoencoder**
2. **Task 2: Variational Autoencoder**
3. **Task 3: Transformer-based Music Generator**
4. **Task 4: Prototype RLHF / Human Preference Tuning**

The project uses MIDI files instead of raw audio. MIDI is suitable for this project because it stores symbolic musical information such as pitch, velocity, note start time, and note end time. This makes it easier to model musical structure compared to raw waveform audio.

---

## 2. GitHub Repository

GitHub Repository Link:

https://github.com/mabishaaaa/CSE425


output:
https://drive.google.com/drive/folders/1ukuuEtT52yYaC9l4T0s2pAQNU23vYT6z?usp=sharing


## 3. Dataset

### 3.1 Dataset Used

Dataset name:

```text
MAESTRO v3.0.0 MIDI Dataset
```

Official dataset source:

```text
https://magenta.tensorflow.org/datasets/maestro
```

Dataset link used for this project:

```text
PASTE_DATASET_LINK_HERE
```

The MAESTRO dataset contains classical piano performances recorded using Yamaha Disklavier pianos. Each MIDI file contains symbolic musical information such as pitch, note onset, note offset, and velocity.

### 3.2 Dataset Structure

Expected dataset folder structure:

```text
maestro-v3.0.0/
├── maestro-v3.0.0.csv
├── 2004/
├── 2006/
├── 2008/
├── 2009/
├── 2011/
├── 2013/
├── 2014/
├── 2015/
└── 2017/
```

The CSV file contains important metadata such as:

```text
canonical_composer
canonical_title
split
year
duration
midi_filename
```

### 3.3 Dataset Split

The official MAESTRO train, validation, and test split is preserved.

The split is used as follows:

```text
Training split     → model training
Validation split   → loss/perplexity monitoring
Test/generated set → final evaluation and comparison
```

The dataset is **not randomly re-split**, because random splitting can cause data leakage between train and test sets.

### 3.4 Dataset Upload Note

The full MAESTRO dataset is **not uploaded to GitHub** because it is large. Users should download the dataset from the official MAESTRO link or use the dataset link provided above.

---

## 4. Generated MIDI Outputs

Generated MIDI files are stored separately in Google Drive because the output folder contains many MIDI files.

Google Drive link for generated MIDI files:

https://drive.google.com/drive/folders/1ukuuEtT52yYaC9l4T0s2pAQNU23vYT6z?usp=sharing

The Google Drive folder contains:

```text
Task 1 LSTM Autoencoder generated MIDI samples
Task 2 VAE generated MIDI samples
Task 2 VAE latent interpolation MIDI samples
Task 3 Transformer generated MIDI samples
Task 4 pretrained Transformer MIDI samples
Task 4 RLHF/prototype fine-tuned MIDI samples
Random Generator baseline MIDI samples
Markov Chain baseline MIDI samples
```

Suggested Google Drive structure:

```text
Generated MIDI Outputs/
├── Task 1 - LSTM Autoencoder/
├── Task 2 - VAE Samples/
├── Task 2 - VAE Interpolation/
├── Task 3 - Transformer/
├── Task 4 - RLHF Prototype/
└── Baselines/
```

---

## 5. Report

The final IEEE-style project report is included in the repository.

Report files:

```text
report/main.tex
report/report.pdf
```

Google Drive report link, if applicable:

```text
PASTE_GOOGLE_DRIVE_REPORT_LINK_HERE
```

The report includes:

```text
Abstract
Introduction
Methodology
Result Analysis
Conclusion
References
```

---

## 6. Project Pipeline

The overall system pipeline is:

```text
MAESTRO MIDI Dataset
        ↓
Exploratory Data Analysis
        ↓
MIDI Preprocessing
        ↓
Binary Piano-Roll Windows / Symbolic Event Tokens
        ↓
Task 1: LSTM Autoencoder
Task 2: Variational Autoencoder
Task 3: Transformer Generator
Task 4: Prototype RLHF / Preference Tuning
        ↓
Generated MIDI Outputs
        ↓
Metric Evaluation and Result Analysis
```

---

## 7. Exploratory Data Analysis

Exploratory Data Analysis, or EDA, is performed before model training to understand the dataset.

EDA includes:

```text
Dataset split distribution
Duration distribution
Duration distribution by split
Top composer distribution
Optional pitch distribution
Optional velocity distribution
Optional note count distribution
Optional piano-roll sparsity analysis
```

EDA helps justify preprocessing decisions. Since MAESTRO pieces are long, they are divided into shorter fixed-length windows. Since MAESTRO is mainly classical piano, multi-genre generation is approximated using composer or style-period proxy labels.

Generated EDA plots include:

```text
split_distribution.png
duration_histogram.png
duration_by_split_boxplot.png
top_composers.png
```

---

## 8. Preprocessing

### 8.1 Piano-Roll Preprocessing for Task 1 and Task 2

For the LSTM Autoencoder and VAE, MIDI files are converted into binary piano-roll windows.

Steps:

```text
1. Load MIDI file using pretty_midi.
2. Extract piano-roll at 16 frames per second.
3. Keep only MIDI pitches 21 to 108.
4. Transpose piano-roll to shape (T, 88).
5. Binarize the piano-roll:
   active note = 1
   silence = 0
6. Split the piano-roll into 128-step windows.
7. Filter out sparse windows with less than 2% active cells.
```

Final input shape:

```text
(128, 88)
```

At 16 frames per second:

```text
128 time steps = 8 seconds of music
```

### 8.2 Token Preprocessing for Task 3 and Task 4

For the Transformer and RLHF prototype, MIDI is represented using symbolic event tokens.

The implemented token types include:

```text
Note-On
Note-Off
Time-Shift
Velocity
```

The supplementary guide recommends REMI tokenization using MidiTok. This implementation uses a custom symbolic event-token representation while preserving the same autoregressive next-token prediction objective.

---

## 9. Models Implemented

## 9.1 Task 1: LSTM Autoencoder

### Purpose

The LSTM Autoencoder learns a compressed latent representation of binary piano-roll windows and reconstructs the original input sequence.

### Architecture

```text
Input piano-roll window: (128, 88)
        ↓
LSTM Encoder
        ↓
Final hidden state
        ↓
Linear projection
        ↓
Latent vector z: 64 dimensions
        ↓
Repeat latent vector across time steps
        ↓
LSTM Decoder
        ↓
Reconstructed piano-roll: (128, 88)
```

### Important Details

```text
Uses final LSTM hidden state as latent representation
Latent dimension = 64
Uses raw logits during training
Uses sigmoid only during MIDI generation
Uses Focal Loss with positive class weighting
Generates 5 MIDI samples
```

### Main Outputs

```text
task1_sample_1.mid
task1_sample_2.mid
task1_sample_3.mid
task1_sample_4.mid
task1_sample_5.mid
loss_curve_task1.png
ae_weights.pt
```

---

## 9.2 Task 2: Variational Autoencoder

### Purpose

The VAE extends the Autoencoder by learning a probabilistic latent space. It supports random sampling and latent interpolation.

### Architecture

```text
Input piano-roll window: (128, 88)
        ↓
LSTM Encoder
        ↓
Mean vector μ
Log-variance vector logσ²
        ↓
Reparameterization trick
z = μ + σ × ε
        ↓
LSTM Decoder
        ↓
Generated / reconstructed piano-roll
```

### Loss Function

The VAE uses:

```text
VAE Loss = Reconstruction Loss + β × KL Divergence
```

KL annealing is used to gradually increase β from 0 to 1. This reduces the risk of posterior collapse.

### Multi-Genre Proxy

MAESTRO is mainly classical piano, so true multi-genre generation is limited. The implementation uses composer or style-period proxy labels:

```text
Baroque
Classical
Romantic
Modern
```

### Main Outputs

```text
task2_sample_1.mid
task2_sample_2.mid
task2_sample_3.mid
task2_sample_4.mid
task2_sample_5.mid
task2_sample_6.mid
task2_sample_7.mid
task2_sample_8.mid

task2_interp_1_alpha0.00.mid
task2_interp_2_alpha0.14.mid
task2_interp_3_alpha0.29.mid
task2_interp_4_alpha0.43.mid
task2_interp_5_alpha0.57.mid
task2_interp_6_alpha0.71.mid
task2_interp_7_alpha0.86.mid
task2_interp_8_alpha1.00.mid

loss_curve_task2.png
vae_weights.pt
```

---

## 9.3 Task 3: Transformer Generator

### Purpose

The Transformer models symbolic MIDI event sequences autoregressively. It predicts the next token based on previous tokens.

### Architecture

```text
Symbolic token sequence
        ↓
Token embedding
        ↓
Genre embedding
        ↓
Positional encoding
        ↓
Causal Transformer layers
        ↓
Output projection to vocabulary size
        ↓
Next-token prediction
```

### Training Objective

The model is trained using shifted input-target pairs:

```text
Input:  x1, x2, ..., xT-1
Target: x2, x3, ..., xT
```

The Transformer uses:

```text
Cross-entropy loss
Causal masking
Validation perplexity
Top-k sampling
Temperature scaling
```

### Main Outputs

```text
task3_Baroque_1.mid
task3_Classical_2.mid
task3_Romantic_3.mid
task3_Modern_4.mid
task3_Baroque_5.mid
task3_Classical_6.mid
task3_Romantic_7.mid
task3_Modern_8.mid
task3_Baroque_9.mid
task3_Classical_10.mid

metrics_task3.png
comparison_task3.png
transformer_weights.pt
```

---

## 9.4 Task 4: Prototype RLHF / Human Preference Tuning

### Purpose

Task 4 introduces a prototype preference-based fine-tuning workflow. The goal is to guide the Transformer generator toward higher-scoring outputs using a reward model.

### Pipeline

```text
Trained Transformer Generator
        ↓
Generate pretrained MIDI samples
        ↓
Reward model scores generated samples
        ↓
Fine-tune generator using reward signal
        ↓
Generate RLHF/prototype fine-tuned MIDI samples
```

### Important Note

Task 4 is implemented as a prototype preference-tuning workflow. It demonstrates the RLHF structure but should not be interpreted as a full large-scale RLHF experiment unless real human feedback is collected and used directly.

A complete RLHF implementation would require:

```text
Real listener ratings from at least 10 participants
Reward model trained directly on those ratings
Policy-gradient update using stored token log probabilities
Before/after human listening evaluation
```

### Main Outputs

```text
task4_pretrained_1.mid
task4_pretrained_2.mid
task4_pretrained_3.mid
task4_pretrained_4.mid
task4_pretrained_5.mid
task4_pretrained_6.mid
task4_pretrained_7.mid
task4_pretrained_8.mid
task4_pretrained_9.mid
task4_pretrained_10.mid

task4_rlhf_1.mid
task4_rlhf_2.mid
task4_rlhf_3.mid
task4_rlhf_4.mid
task4_rlhf_5.mid
task4_rlhf_6.mid
task4_rlhf_7.mid
task4_rlhf_8.mid
task4_rlhf_9.mid
task4_rlhf_10.mid

rl_training_task4.png
reward_model.pt
rlhf_generator.pt
task4_metrics.csv
```

---

## 10. Baseline Models

Two baseline models are implemented.

## 10.1 Random Generator

The Random Generator samples:

```text
Random pitch from MIDI range 21 to 108
Random duration from a fixed duration set
Random velocity
```

This baseline does not learn from the dataset and serves as a lower-bound comparison.

## 10.2 Markov Chain Model

The Markov Chain model learns first-order pitch transitions from training MIDI files:

```text
current pitch → next pitch probability distribution
```

It captures local melodic transitions but does not model long-range musical structure.

### Baseline Outputs

```text
baseline_random_1.mid
baseline_random_2.mid
baseline_random_3.mid

baseline_markov_1.mid
baseline_markov_2.mid
baseline_markov_3.mid
```

---

## 11. Evaluation Metrics

The project evaluates generated MIDI files using the following metrics.

## 11.1 Pitch Histogram Similarity

Measures the difference between pitch-class distributions of generated MIDI and reference MIDI.

```text
Lower value = generated music is closer to real music in pitch-class usage
```

## 11.2 Rhythm Diversity

Measures how many unique note durations appear relative to the total number of notes.

```text
Higher value = more rhythmic variety
```

## 11.3 Repetition Ratio

Measures repeated 4-note pitch patterns.

```text
Moderate repetition = musical structure
Very high repetition = possible degenerate looping
Very low repetition = possible lack of structure
```

## 11.4 Perplexity

Used for the Transformer model.

```text
Lower perplexity = better next-token prediction
```

## 11.5 Preference Score

Used for the Task 4 prototype preference-tuning stage.

```text
Higher score = higher reward/preference estimate
```

---

## 12. Result Summary

## 12.1 EDA Results

EDA showed that:

```text
MAESTRO is dominated by classical piano composers.
The training split is the largest split.
Most performances are several minutes long.
Window-based preprocessing is necessary.
Composer/style-period proxy labels are needed for the multi-genre requirement.
```

## 12.2 Task 1 Result

The LSTM Autoencoder showed stable convergence. Both training and validation loss decreased smoothly.

Main evidence:

```text
loss_curve_task1.png
```

## 12.3 Task 2 Result

The VAE generated valid samples and interpolation outputs. However, KL divergence dropped sharply during training, suggesting possible partial posterior collapse.

Main evidence:

```text
loss_curve_task2.png
```

## 12.4 Task 3 Result

The Transformer achieved decreasing training and validation loss. Validation perplexity also decreased. It achieved higher rhythm diversity than the Random Generator and Markov Chain baselines.

Main evidence:

```text
metrics_task3.png
comparison_task3.png
```

## 12.5 Task 4 Result

The prototype RLHF/preference-tuning pipeline generated before and after MIDI outputs and produced a reward progression plot.

Main evidence:

```text
rl_training_task4.png
```

---

## 13. Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── report/
│   ├── main.tex
│   └── report.pdf
├── codes/
│   ├── eda.py
│   ├── task1.py
│   ├── task2.py
│   ├── task3_transformer.py
│   ├── task4_rlhf.py
│   ├── baselines.py
│   ├── run_baselines.py
│   ├── evaluate_all.py
│   ├── generate.py
│   ├── generate_vae.py
│   ├── generate_transformer.py
│   ├── generate_rlhf.py
│   ├── train.py
│   ├── train_vae.py
│   ├── train_transformer.py
│   ├── train_rlhf.py
│   ├── reward_model.py
│   ├── train_reward_model.py
│   └── prepare_feedback_data.py
├── plots/
│   ├── split_distribution.png
│   ├── duration_histogram.png
│   ├── duration_by_split_boxplot.png
│   ├── top_composers.png
│   ├── loss_curve_task1.png
│   ├── loss_curve_task2.png
│   ├── metrics_task3.png
│   ├── comparison_task3.png
│   └── rl_training_task4.png
└── sample_outputs/
    └── optional_small_samples/
```

---

## 14. File Descriptions

### `README.md`

Main documentation file. It explains the project overview, dataset, pipeline, models, outputs, file descriptions, running instructions, limitations, and group contributions.

### `requirements.txt`

Contains the Python packages required to run the project.

### `.gitignore`

Prevents large or unnecessary files from being uploaded to GitHub, such as virtual environments, datasets, model weights, generated audio, and large MIDI folders.

---

## 14.1 Report Files

### `report/main.tex`

LaTeX source code for the IEEE-style project report.

### `report/report.pdf`

Compiled project report in PDF format.


## 15. How to Run the Project

### Step 1: Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download and extract MAESTRO v3.0.0

Download from:

```text
https://magenta.tensorflow.org/datasets/maestro
```

Place it in the project root:

```text
maestro-v3.0.0/
```

### Step 4: Run EDA

```bash
python codes/eda.py
```

If paths are different:

```bash
python codes/eda.py \
  --csv-path "path/to/maestro-v3.0.0/maestro-v3.0.0.csv" \
  --midi-root "path/to/maestro-v3.0.0" \
  --base-dir "."
```

### Step 5: Run Task 1

```bash
python codes/task1.py
```

### Step 6: Run Task 2

```bash
python codes/task2.py
```

### Step 7: Run Task 3

```bash
python codes/task3_transformer.py
```

### Step 8: Run Task 4

```bash
python codes/task4_rlhf.py
```

### Step 9: Run baselines

```bash
python codes/run_baselines.py
```

### Step 10: Run evaluation

```bash
python codes/evaluate_all.py
```

---

## 16. Dependencies

Main dependencies:

```text
torch
numpy
pandas
matplotlib
pretty_midi
tqdm
```

Optional dependencies:

```text
miditok
music21
```

Example `requirements.txt`:

```text
torch
numpy
pandas
matplotlib
pretty_midi
tqdm
miditok
music21
```

---

## 17. Notes on Large Files

The following files and folders are not included in the GitHub repository:

```text
Full MAESTRO dataset
Large generated MIDI folders
Model weight files
Virtual environment folders
Audio exports such as WAV or MP3
```

These should be excluded using `.gitignore`.

Generated MIDI files are shared through Google Drive.

---

## 18. Limitations

This project has several limitations:

1. **MAESTRO is classical-only**  
   The dataset mainly contains classical piano performances. Multi-genre generation is approximated using composer or style-period proxy labels.

2. **Binary piano-roll removes dynamics**  
   Tasks 1 and 2 use binary piano-rolls, which remove velocity and expressive dynamics.

3. **Custom Transformer tokenization**  
   The supplementary guide recommends REMI tokenization using MidiTok. This implementation uses a custom symbolic event-token representation.

4. **Small-scale training**  
   Some models use reduced dataset subsets or fewer epochs due to compute limitations.

5. **Prototype RLHF**  
   Task 4 demonstrates the structure of a preference-tuning workflow but is not a full large-scale RLHF experiment.

6. **Generated music quality varies**  
   Some generated outputs may be simple, repetitive, or musically inconsistent, which is expected for small-scale symbolic music generation.

---

## 19. Group Member Contributions

| Group Member | Contribution |
|---|---|
| Maliha Binte Shamim | EDA, preprocessing, report writing, Task 1 LSTM Autoencoder implementation and generated MIDI outputs, Task 2 VAE implementation and latent interpolation, Plot generation, MIDI organization, README, and presentation preparation |
| Muntasrir Mahmud    | Task 3 Transformer implementation and perplexity evaluation, Task 4 RLHF/prototype preference-tuning pipeline,  Baseline models and evaluation metrics|

## 20. Presentation Notes

Recommended presentation flow:

```text
1. Project overview
2. EDA
3. Preprocessing
4. Task 1 architecture and result
5. Play Task 1 generated MIDI
6. Task 2 architecture and result
7. Play Task 2 generated MIDI
8. Task 2 interpolation demo
9. Task 3 Transformer result
10. Task 4 prototype preference-tuning result
11. Conclusion and limitations
```

Recommended MIDI files to play:

```text
task1_sample_1.mid
task2_sample_1.mid
task2_interp_1_alpha0.00.mid
task2_interp_4_alpha0.43.mid
task2_interp_8_alpha1.00.mid
task3_Baroque_1.mid
task4_rlhf_1.mid
```

---

## 21. References

1. MAESTRO Dataset  
   https://magenta.tensorflow.org/datasets/maestro

2. pretty_midi  
   https://github.com/craffel/pretty-midi

3. PyTorch  
   https://pytorch.org/

4. MidiTok  
   https://github.com/Natooz/MidiTok

5. Transformer Architecture  
   Vaswani et al., "Attention Is All You Need", 2017.

6. Variational Autoencoder  
   Kingma and Welling, "Auto-Encoding Variational Bayes", 2014.

---

## 22. Final Submission Checklist

Before submitting, make sure:

```text
[ ] GitHub repository is public
[ ] README.md is complete
[ ] README introduces each file clearly
[ ] Dataset link is added
[ ] Google Drive MIDI link is added
[ ] Google Drive report link is added if needed
[ ] Report includes the Google Drive MIDI link
[ ] Groupmate contributions are listed
[ ] Code files are commented
[ ] Report PDF is included
[ ] LaTeX source is included
[ ] Plots are included
[ ] Full MAESTRO dataset is not uploaded
[ ] Virtual environment is not uploaded
[ ] Large model weights are not uploaded unless required
[ ] Generated MIDI files are shared through Google Drive
```

---

## 23. License / Academic Use

This project is submitted for academic purposes as part of CSE425 coursework.

```text
Academic use only.
```
