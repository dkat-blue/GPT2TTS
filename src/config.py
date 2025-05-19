# src/config.py
"""
Centralized configuration settings for the TTS model.
Includes hyperparameters, path definitions, and other constants.
"""
import torch
import datetime
import os

# --- Paths and Directories ---
DATA_DIR = "../data/LJSpeech-1.1/" # Base directory for the LJSpeech dataset.
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv") # Path to metadata file (audio clips & transcripts).
WAVS_DIR = os.path.join(DATA_DIR, "wavs") # Directory containing WAV audio files.

# --- Output Directory Setup (Timestamped for each run) ---
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Unique ID for the current training run.
OUTPUT_DIR_BASE = "../training_runs" # Base directory for all training run outputs.
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, f"training_run_{RUN_ID}") # Specific directory for this run.

AUDIO_LOGGING_DIR = os.path.join(OUTPUT_DIR, "training_audio_samples") # For generated audio samples during training.
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints") # For model checkpoints.
BEST_MODEL_VAL_CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "best_val_loss_model.pth") # Checkpoint with best validation loss.
LATEST_MODEL_CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "latest_model_checkpoint.pth") # Latest saved model checkpoint.
TRAIN_LOSS_CHART_FILE = os.path.join(OUTPUT_DIR, "train_loss_chart.png") # Training loss plot.
VAL_LOSS_CHART_FILE = os.path.join(OUTPUT_DIR, "val_loss_chart.png") # Validation loss plot.
TRAINING_LOG_FILE = os.path.join(OUTPUT_DIR, "training_summary_log.txt") # Text log for training run summary.

# --- Dataset and Model Parameters ---
TARGET_SAMPLE_RATE = 24000  # Target audio sampling rate (Hz). Encodec model expects 24kHz.
NUM_AUDIO_CODEBOOKS_EXPECTED = 32 # Expected number of Encodec codebooks.

# Number of samples to use from the dataset.
# Set to an integer (e.g., 1310 for ~10% of LJSpeech) or None for the entire dataset.
MAX_TOTAL_SAMPLES = None

VALIDATION_SPLIT_RATIO = 0.05  # Proportion of data for validation.
RANDOM_SEED_DATASET_SPLIT = 42 # Seed for reproducible train/validation splits.

# --- DataLoader Parameters ---
NUM_DATALOADER_WORKERS = 4  # Number of worker processes for data loading.
PIN_MEMORY_DATALOADER = True # If True, speeds up CUDA transfers by copying tensors to pinned memory.

# --- Training Hyperparameters ---
NUM_EPOCHS = 1  # Number of training epochs.
LEARNING_RATE = 2e-5 # Initial learning rate for AdamW.
BATCH_SIZE = 32    # Samples per batch.
MAX_GRAD_NORM = 1.0 # Max norm for gradient clipping.
WEIGHT_DECAY = 0.01 # Weight decay for AdamW (L2 regularization).
LABEL_SMOOTHING = 0.1 # Label smoothing factor for CrossEntropyLoss.

# --- Learning Rate Scheduler ---
LR_SCHEDULER_TYPE = "linear_with_warmup" # Type of LR scheduler.
NUM_WARMUP_STEPS = 200 # Steps for LR to linearly increase to LEARNING_RATE.

# --- Early Stopping ---
EARLY_STOPPING_PATIENCE = 7  # Epochs with no validation improvement before stopping.
MIN_DELTA_IMPROVEMENT = 0.001 # Minimum change in val loss to be considered an improvement.

# --- Curriculum Learning (Audio Token Length) ---
# Gradually increases max audio sequence length during training.
USE_CURRICULUM_LEARNING = True
CURRICULUM_INITIAL_AUDIO_TOKEN_LEN = 1024 # Initial max audio tokens per sample.
CURRICULUM_INCREMENT_AUDIO_TOKEN_LEN = 128 # Tokens to increase by at each interval.
CURRICULUM_INCREMENT_INTERVAL_EPOCHS = 1 # Frequency (epochs) to increase audio token length.

# --- Audio Logging during Training ---
# Prompts for generating sample audio during training.
LOGGING_SAMPLE_TEXTS = [
    "Hello world, this is a speech synthesis test.",
    "The quick brown fox jumps over the lazy dog.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
]
LOGGING_MAX_NEW_TOKENS = 512 # Max new audio tokens for logged samples.

# --- Device ---
# Selects CUDA if available, otherwise CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- GPT-2 Configuration for TTS ---
GPT2_MODEL_NAME = 'gpt2' # Pretrained GPT-2 model name.

# --- Encodec Configuration ---
ENCODEC_MODEL_NAME = "facebook/encodec_24khz" # Pretrained Encodec model name.
ENCODEC_BANDWIDTH_TRAIN = 24.0 # Target bandwidth for Encodec during training tokenization (kbps).

def create_run_directories():
    """Creates necessary output directories for the current training run."""
    abs_output_dir = os.path.abspath(OUTPUT_DIR)
    abs_audio_logging_dir = os.path.abspath(AUDIO_LOGGING_DIR)
    abs_checkpoint_dir = os.path.abspath(CHECKPOINT_DIR)

    os.makedirs(abs_output_dir, exist_ok=True)
    os.makedirs(abs_audio_logging_dir, exist_ok=True)
    os.makedirs(abs_checkpoint_dir, exist_ok=True)
    print(f"Created output directories for RUN_ID: {RUN_ID} in {abs_output_dir}")

if __name__ == '__main__':
    # Executed if the script is run directly (e.g., for testing or printing config).
    create_run_directories()
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    print(f"Label Smoothing: {LABEL_SMOOTHING}")
    print(f"Best model will be saved to: {os.path.abspath(BEST_MODEL_VAL_CHECKPOINT_FILE)}")
    print(f"Data directory configured as: {os.path.abspath(DATA_DIR)}")
    print(f"MAX_TOTAL_SAMPLES set to: {MAX_TOTAL_SAMPLES if MAX_TOTAL_SAMPLES is not None else 'All'}")
    print(f"Using Curriculum Learning: {USE_CURRICULUM_LEARNING}")
    if USE_CURRICULUM_LEARNING:
        print(f"  Initial Audio Token Length: {CURRICULUM_INITIAL_AUDIO_TOKEN_LEN}")
        print(f"  Increment Interval: {CURRICULUM_INCREMENT_INTERVAL_EPOCHS} epoch(s)")
        print(f"  Increment Amount: {CURRICULUM_INCREMENT_AUDIO_TOKEN_LEN}")
    print(f"Number of DataLoader workers: {NUM_DATALOADER_WORKERS}")
    print(f"Number of Warmup Steps: {NUM_WARMUP_STEPS}")
    print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
