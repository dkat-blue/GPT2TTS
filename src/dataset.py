# src/dataset.py
"""
Handles LJSpeech dataset loading, preprocessing, and splitting.
"""
import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Optional, List, Dict, Tuple # Added Optional, List, Dict, Tuple for type hints

import config # Project-specific configuration

class LJSpeechDataset(Dataset):
    """
    PyTorch Dataset for LJSpeech. Loads audio waveforms and normalized text.
    Audio is resampled to a target sample rate.
    """
    def __init__(self, metadata_df: pd.DataFrame, wavs_dir: str, 
                 processor, target_sample_rate: int, tokenizer=None):
        """
        Initializes the dataset.

        Args:
            metadata_df: DataFrame with 'filename_base' and 'normalized_text'.
            wavs_dir: Directory of WAV audio files.
            processor: Encodec processor (passed for consistency, not used directly here).
            target_sample_rate: Target audio sample rate (Hz).
            tokenizer: GPT-2 tokenizer (passed for consistency).
        """
        self.wavs_dir = wavs_dir
        self.metadata = metadata_df.reset_index(drop=True)
        self.processor = processor 
        self.target_sample_rate = target_sample_rate
        self.tokenizer = tokenizer 

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Retrieves a single sample (audio waveform and text).
        Attempts to load next valid sample on error, wrapping around dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Dict with "text", "audio_waveform" (NumPy array), "sampling_rate".

        Raises:
            RuntimeError: If no valid sample can be loaded after multiple attempts.
        """
        for i in range(len(self.metadata)):
            current_idx = (idx + i) % len(self.metadata)
            try:
                wav_filename_base = self.metadata.iloc[current_idx]['filename_base']
                text = self.metadata.iloc[current_idx]['normalized_text']
                wav_path = os.path.join(self.wavs_dir, wav_filename_base + ".wav")

                waveform, sample_rate = torchaudio.load(wav_path)

                if waveform.numel() == 0: # Skip empty waveform
                    if i == len(self.metadata) -1: raise RuntimeError(f"Exhausted retries: Empty waveform for {wav_path}")
                    continue

                if waveform.shape[0] > 1: # Convert to mono
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                if sample_rate != self.target_sample_rate: # Resample if needed
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                    waveform = resampler(waveform)
                
                if waveform.numel() == 0: # Skip if empty after resampling
                    if i == len(self.metadata) -1: raise RuntimeError(f"Exhausted retries: Empty after resampling for {wav_path}")
                    continue

                audio_sample_np = waveform.squeeze().numpy()
                if audio_sample_np.size == 0: # Skip if NumPy array is empty
                    if i == len(self.metadata) -1: raise RuntimeError(f"Exhausted retries: Zero-size numpy array for {wav_path}")
                    continue
                
                return {"text": text, "audio_waveform": audio_sample_np, "sampling_rate": self.target_sample_rate}

            except Exception as e:
                if i == len(self.metadata) -1: 
                    raise RuntimeError(f"Could not load any valid sample. Last error for {wav_path}: {e}")
                continue 
        
        raise RuntimeError("Exhausted dataset without finding a valid sample.")

def load_and_split_metadata(metadata_file_path: str, wavs_dir: str, 
                            val_ratio: float, random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads metadata, optionally subsamples based on config.MAX_TOTAL_SAMPLES,
    and splits into training and validation DataFrames.

    Args:
        metadata_file_path: Path to 'metadata.csv'.
        wavs_dir: Path to WAV files directory.
        val_ratio: Proportion for validation set.
        random_seed: Seed for reproducible splits/sampling.

    Returns:
        Tuple (train_df, val_df).
    """
    if not os.path.exists(metadata_file_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file_path}")
    if not os.path.exists(wavs_dir):
        raise FileNotFoundError(f"WAVs directory not found: {wavs_dir}")

    try:
        full_metadata_df = pd.read_csv(
            metadata_file_path, sep='|', header=None, 
            names=['filename_base', 'original_text', 'normalized_text'], quoting=3
        )
    except Exception as e:
        raise ValueError(f"Error reading metadata file {metadata_file_path}. Error: {e}")

    samples_to_use_df = full_metadata_df
    num_total_available = len(full_metadata_df)

    if config.MAX_TOTAL_SAMPLES is not None:
        if 0 < config.MAX_TOTAL_SAMPLES < num_total_available:
            print(f"Sampling {config.MAX_TOTAL_SAMPLES} from {num_total_available} entries.")
            samples_to_use_df = full_metadata_df.sample(n=config.MAX_TOTAL_SAMPLES, random_state=random_seed).reset_index(drop=True)
        elif config.MAX_TOTAL_SAMPLES <= 0:
            raise ValueError("MAX_TOTAL_SAMPLES must be positive if set.")
        else: 
            print(f"Using all {num_total_available} entries (MAX_TOTAL_SAMPLES >= dataset size or not limiting).")
    else: 
        print(f"Using full dataset of {num_total_available} entries (MAX_TOTAL_SAMPLES is None).")

    if val_ratio > 0 and len(samples_to_use_df) > 1:
        num_val_samples = max(1, int(len(samples_to_use_df) * val_ratio))
        if num_val_samples >= len(samples_to_use_df): # Ensure training set is not empty
            num_val_samples = max(0, len(samples_to_use_df) - 1)

        if num_val_samples > 0:
            train_df, val_df = train_test_split(
                samples_to_use_df, test_size=num_val_samples, 
                random_state=random_seed, shuffle=True
            )
            print(f"Split: {len(train_df)} train, {len(val_df)} val samples.")
        else:
            train_df = samples_to_use_df
            val_df = pd.DataFrame(columns=samples_to_use_df.columns)
            print(f"Using all {len(train_df)} for training (not enough data for val_ratio={val_ratio}).")
    else:
        train_df = samples_to_use_df
        val_df = pd.DataFrame(columns=samples_to_use_df.columns) 
        print(f"Using all {len(train_df)} for training (no validation split).")
    
    return train_df, val_df

def get_dataloaders(processor, gpt2_tokenizer, 
                    val_ratio: float = config.VALIDATION_SPLIT_RATIO, 
                    batch_size: int = config.BATCH_SIZE) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    Creates DataLoaders for training and validation.

    Args:
        processor: Encodec processor.
        gpt2_tokenizer: GPT-2 tokenizer.
        val_ratio: Proportion for validation.
        batch_size: Batch size.

    Returns:
        Tuple (train_dataloader, val_dataloader). val_dataloader can be None.
        Returns (None, None) if no training data.
    """
    train_df, val_df = load_and_split_metadata(
        config.METADATA_FILE, config.WAVS_DIR,
        val_ratio, config.RANDOM_SEED_DATASET_SPLIT
    )

    if train_df.empty:
        print("Warning: No training data. Cannot create DataLoaders.")
        return None, None

    train_dataset = LJSpeechDataset(
        train_df, config.WAVS_DIR, processor,
        config.TARGET_SAMPLE_RATE, gpt2_tokenizer
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=config.NUM_DATALOADER_WORKERS, 
        pin_memory=config.PIN_MEMORY_DATALOADER if config.DEVICE.type == 'cuda' else False,
        drop_last=len(train_dataset) > batch_size # Drop last if incomplete
    )

    val_dataloader = None
    if not val_df.empty:
        val_dataset = LJSpeechDataset(
            val_df, config.WAVS_DIR, processor,
            config.TARGET_SAMPLE_RATE, gpt2_tokenizer
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=config.NUM_DATALOADER_WORKERS,
            pin_memory=config.PIN_MEMORY_DATALOADER if config.DEVICE.type == 'cuda' else False
        )
    
    return train_dataloader, val_dataloader

def collate_fn(batch: List[Dict[str, any]]) -> Dict[str, List[any]]:
    """
    Custom collate function. Aggregates texts and audio waveforms from batch items.
    """
    texts = [item['text'] for item in batch]
    audio_waveforms = [item['audio_waveform'] for item in batch] 
    return {"texts": texts, "audio_waveforms": audio_waveforms}
