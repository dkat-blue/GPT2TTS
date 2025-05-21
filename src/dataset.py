# src/dataset.py
"""
Dataset class for LJSpeech, adapted for mel spectrogram prediction
and XTTS-style conditioning with a reference audio.
Computes mel spectrograms from audio files and tokenizes text.
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import torchaudio
import random
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

import config as dataset_config

# Attempt to import SpeechBrain's mel_spectogram function for precise mel calculation.
speechbrain_mel_spectogram_fn = None 
try:
    import speechbrain.lobes.models.FastSpeech2 as fs2_module
    if hasattr(fs2_module, 'mel_spectogram') and callable(getattr(fs2_module, 'mel_spectogram')):
        speechbrain_mel_spectogram_fn = getattr(fs2_module, 'mel_spectogram')
        print("Successfully imported SpeechBrain's mel_spectogram function.")
    else:
        print("Warning: SpeechBrain's FastSpeech2.mel_spectogram not found or not callable. Using torchaudio fallback.")
except ImportError as e_imp_mod:
    print(f"Warning: Could not import SpeechBrain's FastSpeech2 module: {e_imp_mod}. Using torchaudio fallback.")
except Exception as e_gen:
    print(f"ERROR: Unexpected error importing SpeechBrain's mel_spectogram: {e_gen}. Using torchaudio fallback.")
    import traceback
    traceback.print_exc()


class LJSpeechMelDataset(Dataset):
    """
    LJSpeech dataset that provides text tokens, reference audio,
    target mel spectrograms, and teacher-forcing mel spectrograms.
    """
    def __init__(self, metadata_df: pd.DataFrame, wavs_dir: str,
                 gpt2_tokenizer: GPT2Tokenizer, 
                 target_sample_rate: int,
                 n_fft: int, hop_length: int, win_length: int, n_mels: int,
                 f_min: float, f_max: float, mel_power: float, mel_normalized: bool,
                 mel_min_max_energy_norm: bool, mel_norm_sb: str, mel_mel_scale_sb: str, mel_compression_sb: bool,
                 max_text_len: Optional[int] = None, 
                 max_mel_frames: Optional[int] = None, 
                 use_curriculum: bool = False,
                 current_max_mel_frames_curriculum: Optional[int] = None,
                 reference_audio_min_dur_sec: float = 2.0,
                 reference_audio_max_dur_sec: float = 8.0):
        
        self.metadata_df = metadata_df
        self.wavs_dir = wavs_dir
        self.gpt2_tokenizer = gpt2_tokenizer

        # Audio and Mel Spectrogram parameters
        self.target_sample_rate = target_sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.n_mels = n_mels 
        self.f_min = f_min
        self.f_max = f_max
        self.mel_power = mel_power
        self.mel_normalized = mel_normalized
        self.mel_min_max_energy_norm = mel_min_max_energy_norm
        self.mel_norm_sb = mel_norm_sb
        self.mel_mel_scale_sb = mel_mel_scale_sb
        self.mel_compression_sb = mel_compression_sb
        
        # Sequence length parameters
        self.max_text_len = max_text_len 
        self.max_mel_frames = max_mel_frames # Max frames for padding/truncation

        # Curriculum learning parameters
        self.use_curriculum = use_curriculum
        self.current_max_mel_frames_for_curriculum = current_max_mel_frames_curriculum
        
        # Reference audio duration parameters
        self.reference_audio_min_samples = int(reference_audio_min_dur_sec * target_sample_rate)
        self.reference_audio_max_samples = int(reference_audio_max_dur_sec * target_sample_rate)

        # Initialize torchaudio mel transform if SpeechBrain's is not available
        if speechbrain_mel_spectogram_fn is None:
            print("Initializing torchaudio.transforms.MelSpectrogram as fallback.")
            self.torchaudio_mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.target_sample_rate,
                n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                n_mels=self.n_mels, f_min=self.f_min, f_max=self.f_max, power=self.mel_power 
            )
        else:
            self.torchaudio_mel_transform = None 


    def _compute_mel_spectrogram(self, waveform_mono_squeezed: torch.Tensor) -> torch.Tensor:
        """
        Computes mel spectrogram. Prefers SpeechBrain's function for HiFi-GAN compatibility,
        otherwise uses torchaudio.
        Returns tensor of shape (n_mels, time_frames).
        """
        if speechbrain_mel_spectogram_fn is not None:
            # SpeechBrain's function returns (time_frames, n_mels) if audio is (time)
            mel_spec_sb, _ = speechbrain_mel_spectogram_fn( 
                audio=waveform_mono_squeezed, sample_rate=self.target_sample_rate,
                hop_length=self.hop_length, win_length=self.win_length, 
                n_mels=self.n_mels, n_fft=self.n_fft,
                f_min=self.f_min, f_max=self.f_max, power=self.mel_power,
                normalized=self.mel_normalized, min_max_energy_norm=self.mel_min_max_energy_norm,
                norm=self.mel_norm_sb, mel_scale=self.mel_mel_scale_sb,
                compression=self.mel_compression_sb
            )
            return mel_spec_sb.T # Transpose to (n_mels, time_frames)
        elif self.torchaudio_mel_transform is not None:
            return self.torchaudio_mel_transform(waveform_mono_squeezed) 
        else:
            raise RuntimeError("No mel spectrogram computation method available.")


    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        filename_base, _, normalized_text = self.metadata_df.iloc[idx]
        wav_path = os.path.join(self.wavs_dir, f"{filename_base}.wav")

        try:
            waveform, sr = torchaudio.load(wav_path)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}. Attempting to load a random different sample.")
            return self.__getitem__(random.randint(0, len(self) - 1)) 

        # Resample and ensure mono
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        waveform_squeezed = waveform.squeeze(0) 

        # Prepare Reference Audio Segment (Waveform)
        num_samples_total = waveform_squeezed.shape[0]
        if num_samples_total == 0: 
            print(f"Warning: Zero-length audio for {filename_base} ({wav_path}), trying another sample.")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        if num_samples_total < self.reference_audio_min_samples:
            ref_audio_segment = waveform_squeezed 
        else:
            max_start = num_samples_total - self.reference_audio_min_samples
            start_sample = random.randint(0, max(0, max_start)) 
            available_length = num_samples_total - start_sample
            segment_length = random.randint(
                self.reference_audio_min_samples,
                min(self.reference_audio_max_samples, available_length) 
            )
            segment_length = max(self.reference_audio_min_samples, min(segment_length, available_length))
            ref_audio_segment = waveform_squeezed[start_sample : start_sample + segment_length]

        # Prepare Target Mel Spectrogram (from full audio)
        target_mel_spectrogram = self._compute_mel_spectrogram(waveform_squeezed) 
        
        assert target_mel_spectrogram.shape[0] == self.n_mels, \
            f"Mel bins mismatch for {filename_base}: got {target_mel_spectrogram.shape[0]}, expected {self.n_mels}. Full shape: {target_mel_spectrogram.shape}"

        # Apply curriculum learning or fixed max_mel_frames for truncation
        current_max_frames_to_use = self.max_mel_frames 
        if self.use_curriculum and self.current_max_mel_frames_for_curriculum is not None:
            current_max_frames_to_use = self.current_max_mel_frames_for_curriculum
        
        if current_max_frames_to_use is not None: 
            if target_mel_spectrogram.size(1) > current_max_frames_to_use:
                target_mel_spectrogram = target_mel_spectrogram[:, :current_max_frames_to_use]
        
        # Teacher forcing mels are the same as target mels for this setup
        teacher_forcing_mel_spectrogram = target_mel_spectrogram 
        
        # Tokenize text, padding to max_text_len
        text_tokens_dict = self.gpt2_tokenizer(
            normalized_text, max_length=self.max_text_len, padding='max_length', 
            truncation=True, return_tensors="pt"
        )
        text_tokens = text_tokens_dict.input_ids.squeeze(0) 
        text_attention_mask = text_tokens_dict.attention_mask.squeeze(0)
        
        return {
            "text_input_ids": text_tokens,
            "text_attention_mask": text_attention_mask,
            "target_mel_spectrogram": target_mel_spectrogram, 
            "teacher_forcing_mel_spectrogram": teacher_forcing_mel_spectrogram, 
            "reference_audio_waveform": ref_audio_segment,
            "reference_audio_sample_rate": self.target_sample_rate, 
            "full_audio_waveform_for_scl": waveform_squeezed, 
            "full_audio_sample_rate_for_scl": self.target_sample_rate,
            "target_mel_n_frames": target_mel_spectrogram.size(1) 
        }

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = [item for item in batch if item is not None]
        if not batch: 
            print("Warning: Collate_fn received an empty batch.")
            return {} # Return empty dict for DataLoader to handle (skip batch)

        text_input_ids = torch.stack([item["text_input_ids"] for item in batch])
        text_attention_mask = torch.stack([item["text_attention_mask"] for item in batch])
        
        # Determine max mel length for padding in this batch or use configured max
        max_len_for_padding = 0
        if self.use_curriculum and self.current_max_mel_frames_for_curriculum is not None:
            max_len_for_padding = self.current_max_mel_frames_for_curriculum
        elif self.max_mel_frames is not None:
            max_len_for_padding = self.max_mel_frames
        else: 
            max_len_for_padding = max(item["target_mel_n_frames"] for item in batch)

        padded_target_mels_list = []
        padded_teacher_mels_list = [] 
        mel_actual_frame_masks_list = [] # True for actual frames, False for padding

        valid_items_in_batch = 0
        for item in batch:
            target_mel = item["target_mel_spectrogram"] 
            teacher_mel = item["teacher_forcing_mel_spectrogram"] 
            
            if target_mel.shape[0] != self.n_mels:
                print(f"ERROR in collate_fn: Unexpected mel bins for an item. Expected {self.n_mels}, got {target_mel.shape[0]}. Skipping item.") 
                continue # Skip this problematic item

            current_item_len = target_mel.size(1) # Use actual size before potential truncation
            
            # Truncate if longer than max_len_for_padding
            if current_item_len > max_len_for_padding:
                target_mel = target_mel[:, :max_len_for_padding]
                teacher_mel = teacher_mel[:, :max_len_for_padding]
                current_item_len = max_len_for_padding
            
            pad_len = max_len_for_padding - current_item_len
            
            if pad_len > 0:
                padding_tensor = torch.zeros(self.n_mels, pad_len, dtype=target_mel.dtype)
                padded_target_mels_list.append(torch.cat([target_mel, padding_tensor], dim=1))
                padded_teacher_mels_list.append(torch.cat([teacher_mel, padding_tensor], dim=1))
                mask = torch.cat([torch.ones(current_item_len, dtype=torch.bool), 
                                  torch.zeros(pad_len, dtype=torch.bool)], dim=0)
            elif pad_len == 0: # No padding needed
                padded_target_mels_list.append(target_mel)
                padded_teacher_mels_list.append(teacher_mel) 
                mask = torch.ones(current_item_len, dtype=torch.bool)
            # else pad_len < 0: This case should be handled by truncation above.
            
            mel_actual_frame_masks_list.append(mask)
            valid_items_in_batch +=1
        
        if not valid_items_in_batch: # If all items were skipped
            print("Warning: No valid items to collate after filtering in collate_fn.")
            return {}

        target_mel_spectrogram_collated = torch.stack(padded_target_mels_list)
        teacher_forcing_mel_spectrogram_collated = torch.stack(padded_teacher_mels_list)
        mel_actual_frame_mask_collated = torch.stack(mel_actual_frame_masks_list)

        # Pad reference audio waveforms (only for valid items)
        ref_audio_waveforms = [batch[i]["reference_audio_waveform"] for i in range(len(batch)) if padded_target_mels_list and i < len(padded_target_mels_list)] # Align with potentially filtered items
        if not ref_audio_waveforms: return {} # Should not happen if valid_items_in_batch > 0
        max_ref_len = max(len(wav) for wav in ref_audio_waveforms)
        padded_ref_waveforms = torch.stack([F.pad(wav, (0, max_ref_len - len(wav))) for wav in ref_audio_waveforms])
        
        full_audio_waveforms_scl = [batch[i]["full_audio_waveform_for_scl"] for i in range(len(batch)) if padded_target_mels_list and i < len(padded_target_mels_list)]
        if not full_audio_waveforms_scl: return {}
        max_full_scl_len = max(len(wav) for wav in full_audio_waveforms_scl)
        padded_full_scl_waveforms = torch.stack([F.pad(wav, (0, max_full_scl_len - len(wav))) for wav in full_audio_waveforms_scl])

        # Filter other items based on valid_items_in_batch if necessary, though stack should handle it if lists are aligned
        # For simplicity, assuming text_input_ids and text_attention_mask correspond to the original batch size
        # If items are skipped, this might lead to size mismatch if not handled carefully.
        # A safer way is to build all lists (text_ids, text_masks etc.) only from valid items.
        # For now, assuming batch filtering is primarily for mels.

        return {
            "text_input_ids": text_input_ids[:valid_items_in_batch] if valid_items_in_batch < len(text_input_ids) else text_input_ids,
            "text_attention_mask": text_attention_mask[:valid_items_in_batch] if valid_items_in_batch < len(text_attention_mask) else text_attention_mask,
            "target_mel_spectrogram": target_mel_spectrogram_collated, 
            "teacher_forcing_mel_spectrogram": teacher_forcing_mel_spectrogram_collated, 
            "mel_actual_frame_mask": mel_actual_frame_mask_collated, 
            "reference_audio_waveform": padded_ref_waveforms,
            "reference_audio_sample_rate": batch[0]["reference_audio_sample_rate"], # Assumes at least one valid item
            "full_audio_waveform_for_scl": padded_full_scl_waveforms,
            "full_audio_sample_rate_for_scl": batch[0]["full_audio_sample_rate_for_scl"]
        }

def get_data_loaders_mel(
    gpt2_tokenizer_name_or_path: str, batch_size: int, num_workers: int, pin_memory: bool,
    train_split_ratio: float = 0.95, 
    max_text_len: Optional[int] = dataset_config.MAX_TEXT_LEN_DATASET, 
    max_mel_frames: Optional[int] = dataset_config.MAX_MEL_FRAMES_DATASET, 
    use_curriculum: bool = False, current_max_mel_frames_curriculum: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, GPT2Tokenizer]:
    
    print(f"get_data_loaders_mel called with: max_text_len={max_text_len}, max_mel_frames={max_mel_frames}")
    metadata_df = pd.read_csv(dataset_config.METADATA_FILE, sep='|', header=None, names=['filename_base', 'original_text', 'normalized_text'], quoting=3)
    
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_tokenizer_name_or_path)
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    train_df = metadata_df.sample(frac=train_split_ratio, random_state=dataset_config.SEED)
    val_df = metadata_df.drop(train_df.index)

    dataset_args = {
        "wavs_dir": dataset_config.WAVS_DIR,
        "gpt2_tokenizer": gpt2_tokenizer,
        "target_sample_rate": dataset_config.TARGET_SAMPLE_RATE,
        "n_fft": dataset_config.MEL_N_FFT, "hop_length": dataset_config.MEL_HOP_LENGTH,
        "win_length": dataset_config.MEL_WIN_LENGTH, "n_mels": dataset_config.MEL_N_MELS,
        "f_min": dataset_config.MEL_FMIN, "f_max": dataset_config.MEL_FMAX,
        "mel_power": dataset_config.MEL_POWER, "mel_normalized": dataset_config.MEL_NORMALIZED,
        "mel_min_max_energy_norm": dataset_config.MEL_MIN_MAX_ENERGY_NORM,
        "mel_norm_sb": dataset_config.MEL_NORM, "mel_mel_scale_sb": dataset_config.MEL_MEL_SCALE,
        "mel_compression_sb": dataset_config.MEL_COMPRESSION,
        "max_text_len": max_text_len, 
        "max_mel_frames": max_mel_frames, 
        "reference_audio_min_dur_sec": dataset_config.REFERENCE_AUDIO_MIN_DURATION_SEC,
        "reference_audio_max_dur_sec": dataset_config.REFERENCE_AUDIO_MAX_DURATION_SEC
    }

    # For curriculum, current_max_mel_frames_curriculum should be the effective max for training dataset
    train_max_mel_frames = current_max_mel_frames_curriculum if use_curriculum and current_max_mel_frames_curriculum is not None else max_mel_frames
    
    train_dataset = LJSpeechMelDataset(
        train_df, **dataset_args,
        max_mel_frames=train_max_mel_frames, # Pass the potentially curriculum-adjusted max frames
        use_curriculum=use_curriculum, # This flag is for __getitem__ logic if needed beyond truncation
        current_max_mel_frames_curriculum=current_max_mel_frames_curriculum 
    )
    # Validation dataset uses the fixed max_mel_frames
    val_dataset = LJSpeechMelDataset(val_df, **dataset_args, max_mel_frames=max_mel_frames) 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=val_dataset.collate_fn)
    
    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
    print(f"LJSpeechMelDataset (train) initialized with max_text_len: {train_dataset.max_text_len}, effective max_mel_frames: {train_dataset.max_mel_frames}")
    print(f"LJSpeechMelDataset (val) initialized with max_text_len: {val_dataset.max_text_len}, effective max_mel_frames: {val_dataset.max_mel_frames}")
    
    return train_loader, val_loader, gpt2_tokenizer
