# src/config.py
"""
Configuration settings for the TTS model, training, and inference.
Defines paths, model parameters, and hyperparameters.
"""
import torch
import os

# --- General Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Determine project root based on this file's location (assuming src/config.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Output directory for all training artifacts, relative to project root
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "training_runs_mel")
LOG_INTERVAL = 100  # Log training progress every N global steps
SAVE_INTERVAL = 1000 # Save checkpoint every N global steps
VALIDATION_INTERVAL = 1000 # Perform validation every N global steps
SEED = 42 # Random seed for reproducibility

# --- Dataset Settings ---
DATA_DIR_NAME = "data" # Name of the data directory at project root
DATASET_NAME = "LJSpeech-1.1" # Name of the dataset subdirectory
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_DIR_NAME, DATASET_NAME)
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
WAVS_DIR = os.path.join(DATA_DIR, "wavs")
TARGET_SAMPLE_RATE = 22050 # Target sample rate for audio processing (Hz), matches HiFi-GAN
NUM_WORKERS = 4             # Number of workers for DataLoader
PIN_MEMORY = True           # Use pinned memory for DataLoader for faster CPU to GPU transfer
TRAIN_SPLIT_RATIO = 0.95    # Proportion of data for training set
REFERENCE_AUDIO_MIN_DURATION_SEC = 2.0 # Minimum duration for reference audio segments (seconds)
REFERENCE_AUDIO_MAX_DURATION_SEC = 8.0 # Maximum duration for reference audio segments (seconds)

# --- GPT-2 Model Settings ---
GPT2_MODEL_NAME = "gpt2"  # Base GPT-2 model identifier from Hugging Face
GPT2_N_POSITIONS = 1024 # Maximum sequence length (context window) for GPT-2

# --- Mel Spectrogram Settings (Aligned with SpeechBrain HiFi-GAN defaults for LJSpeech) ---
# These parameters are critical for compatibility with the pre-trained HiFi-GAN vocoder.
MEL_N_FFT = 1024        # FFT window size
MEL_HOP_LENGTH = 256    # Hop length between frames (for 22050Hz SR, this is ~11.6ms per frame)
MEL_WIN_LENGTH = 1024   # Window length for STFT, typically same as N_FFT
MEL_N_MELS = 80         # Number of Mel frequency bands
MEL_FMIN = 0.0          # Minimum frequency for Mel filterbank (Hz)
MEL_FMAX = 8000.0       # Maximum frequency for Mel filterbank (Hz)
# Additional parameters for SpeechBrain's mel_spectrogram function to match HiFi-GAN's training:
MEL_POWER = 1.0                 # Spectrogram power (1.0 for magnitude, 2.0 for power)
MEL_NORMALIZED = False          # Whether to normalize the STFT output
MEL_MIN_MAX_ENERGY_NORM = True  # Apply min-max energy normalization (SpeechBrain specific)
MEL_NORM = "slaney"             # Type of filterbank normalization (SpeechBrain specific)
MEL_MEL_SCALE = "slaney"        # Mel scale to use (SpeechBrain specific)
MEL_COMPRESSION = True          # Apply dynamic range compression (SpeechBrain specific)


# --- Conditioning Encoder Settings (Processes reference audio mel spectrograms) ---
COND_ENC_INPUT_DIM = MEL_N_MELS # Input dimension is the number of mel bins
COND_ENC_N_ATTN_LAYERS = 6      # Number of Transformer Encoder layers
COND_ENC_N_HEADS = 8            # Number of attention heads in each Transformer layer
COND_ENC_EMBED_DIM = 512        # Internal embedding dimension of the Conditioning Encoder
# Output dimension of Conditioning Encoder, should match GPT-2's hidden size for compatibility
COND_ENC_OUTPUT_DIM = {
    "gpt2": 768, "gpt2-medium": 1024, "gpt2-large": 1280, "gpt2-xl": 1600,
}.get(GPT2_MODEL_NAME, 768)

# --- Perceiver Resampler Settings (Distills Conditioning Encoder output to fixed latents) ---
PERCEIVER_N_LATENTS = 32        # Number of latent vectors to output (fixed-size representation of style)
PERCEIVER_LATENT_DIM = COND_ENC_OUTPUT_DIM # Dimension of latent vectors, matches GPT-2 hidden size
PERCEIVER_N_HEADS = 8           # Number of attention heads in Perceiver's cross-attention
PERCEIVER_N_CROSS_ATTN_LAYERS = 2 # Number of cross-attention layers in Perceiver

# --- Max Lengths for GPT-2 Input Components ---
# These values ensure the total input sequence length does not exceed GPT2_N_POSITIONS.
# PERCEIVER_N_LATENTS + MAX_TEXT_LEN_DATASET + (1 for start_mel_token) + MAX_MEL_FRAMES_DATASET -1 (for teacher forcing) <= GPT2_N_POSITIONS
# Example: 32 (style) + 192 (text) + 1 (start_mel) + 799 (mel_input_frames) = 1024
MAX_TEXT_LEN_DATASET = 192  # Max number of text tokens for dataset processing and model input
MAX_MEL_FRAMES_DATASET = 800 # Max number of mel frames for dataset processing and model input/output

# --- Speaker Encoder Settings (for SCL Proxy) ---
SPEAKER_ENCODER_MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb" # Pre-trained speaker encoder
SPEAKER_EMBED_DIM = 192 # Output dimension of the ECAPA-TDNN speaker encoder

# --- HiFi-GAN Vocoder Settings ---
HIFIGAN_MODEL_SOURCE = "speechbrain/tts-hifigan-ljspeech" # Pre-trained HiFi-GAN model
HIFIGAN_SAVEDIR = os.path.join(PROJECT_ROOT, "pretrained_models", "tts-hifigan-ljspeech") # Cache directory

# --- Training Hyperparameters ---
BATCH_SIZE = 8          # Number of samples per batch
GRADIENT_ACCUMULATION_STEPS = 4 # Accumulate gradients over N steps for larger effective batch size
LEARNING_RATE = 5e-5    # Initial learning rate for AdamW optimizer
WEIGHT_DECAY = 0.01     # Weight decay for AdamW optimizer
ADAM_EPSILON = 1e-8     # Epsilon for AdamW optimizer
NUM_EPOCHS = 20         # Total number of training epochs
LR_SCHEDULER_TYPE = "cosine" # Type of learning rate scheduler ("linear", "cosine", "constant")
WARMUP_STEPS = 500      # Number of warmup steps for the learning rate scheduler
MAX_GRAD_NORM = 1.0     # Maximum norm for gradient clipping

# --- Loss Settings ---
MEL_LOSS_TYPE = "L1"  # Type of loss for mel spectrogram prediction ("L1" or "MSE")
SCL_PROXY_WEIGHT = 0.5 # Weight for the Speaker Consistency Loss (SCL) Proxy
SCL_PROXY_LOSS_TYPE = "cosine" # Type of loss for SCL Proxy ("cosine" or "mse")
# Dimension for projecting style/speaker embeddings for SCL proxy comparison.
# Should match SPEAKER_EMBED_DIM for direct comparison after style projection.
SCL_PROXY_PROJECTION_DIM = SPEAKER_EMBED_DIM 

# --- Curriculum Learning Settings (Optional - applied to max_mel_frames) ---
USE_CURRICULUM_LEARNING = False # Enable/disable curriculum learning
# Starting max mel frames for curriculum (e.g., ~2.3 seconds for 200 frames)
CURRICULUM_START_MAX_MEL_FRAMES = 200 
# Target max mel frames for curriculum, should not exceed MAX_MEL_FRAMES_DATASET
CURRICULUM_END_MAX_MEL_FRAMES = MAX_MEL_FRAMES_DATASET 
# Increase max mel frames every N global training steps
CURRICULUM_INCREMENT_STEPS = 2000       
# Amount to increase max mel frames by at each curriculum step
CURRICULUM_INCREMENT_AMOUNT = 50        

# --- Inference/Generation Settings ---
# Max mel frames to generate during inference, typically same as max trained length
MAX_MEL_FRAMES_TO_GENERATE = MAX_MEL_FRAMES_DATASET 
# Max mel frames for shorter samples during validation logging (e.g., ~3.5 seconds for 300 frames)
LOGGING_MAX_MEL_FRAMES = 300 
# Temperature for sampling (not directly used in current deterministic mel generation, for future use)
GENERATION_TEMPERATURE = 0.8 

# --- GPT-2 Output Dimension for Mels ---
# The GPT-2 model's final layer will output vectors of this size (number of mel bins).
GPT2_OUTPUT_MEL_BINS = MEL_N_MELS
