# src/evaluation.py
"""
Provides functions for evaluating the TTS model, including running inference
and placeholder functions for common TTS metrics.
Adapted for GPT2TTSMelPredictor.
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Any
import jiwer # For CER calculation
import soundfile as sf 
import os
import torchaudio # For loading reference audio in helper

# Assuming these modules are in the same 'src' directory or accessible
import config as inference_config 
from model import GPT2TTSMelPredictor # Import the specific model class
from utils import generate_speech_mel # Core speech generation utility
from transformers import GPT2Tokenizer


def run_single_inference_mel(
    model_checkpoint_path: str,
    text_to_synthesize: str,
    reference_audio_path: str,
    output_dir: str, # Directory to save the output
    output_filename_base: str = "generated_sample",
    device: Optional[torch.device] = None,
    custom_gen_params: Optional[Dict[str, Any]] = None,
    max_mel_frames_override: Optional[int] = None
):
    """
    Loads a model from a checkpoint, generates speech for a single text prompt
    using a reference audio, and saves the output to the specified directory.

    Args:
        model_checkpoint_path (str): Path to the model checkpoint (.pth file).
        text_to_synthesize (str): The text prompt.
        reference_audio_path (str): Path to the reference audio .wav file.
        output_dir (str): Directory where the generated .wav file will be saved.
        output_filename_base (str, optional): Base name for the output .wav file. Defaults to "generated_sample".
        device (Optional[torch.device], optional): Device to run inference on. Defaults to config.DEVICE.
        custom_gen_params (Optional[Dict[str, Any]], optional): Custom parameters for generate_speech_mel. Defaults to None.
        max_mel_frames_override (Optional[int], optional): Override for max_mel_frames. Defaults to config.LOGGING_MAX_MEL_FRAMES.

    Returns:
        Optional[np.ndarray]: The generated waveform as a NumPy array, or None if failed.
    """
    print(f"\n--- Running Single Inference ---")
    print(f"Model Checkpoint: {model_checkpoint_path}")
    print(f"Text: '{text_to_synthesize}'")
    print(f"Reference Audio: {reference_audio_path}")
    print(f"Output Directory: {output_dir}")

    used_device = device if device is not None else inference_config.DEVICE
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load Tokenizer
    gpt2_tokenizer_inf = GPT2Tokenizer.from_pretrained(inference_config.GPT2_MODEL_NAME)
    if gpt2_tokenizer_inf.pad_token is None:
        gpt2_tokenizer_inf.pad_token = gpt2_tokenizer_inf.eos_token
    
    # Load Model
    inference_model = GPT2TTSMelPredictor.from_pretrained_custom(
        tokenizer=gpt2_tokenizer_inf, 
        checkpoint_path=os.path.abspath(model_checkpoint_path), 
        device=used_device
    )
    if inference_model is None:
        print(f"Failed to load model from {model_checkpoint_path}")
        return None
    inference_model.eval()
    print("Model loaded successfully.")

    # Load and preprocess reference audio
    try:
        ref_waveform, ref_sr = torchaudio.load(reference_audio_path)
        if ref_sr != inference_config.TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=ref_sr, new_freq=inference_config.TARGET_SAMPLE_RATE)
            ref_waveform = resampler(ref_waveform)
        if ref_waveform.shape[0] > 1: # Ensure mono
            ref_waveform = torch.mean(ref_waveform, dim=0)
        else:
            ref_waveform = ref_waveform.squeeze(0)
        print(f"Reference audio loaded and processed. Shape: {ref_waveform.shape}")
    except Exception as e:
        print(f"Error loading reference audio {reference_audio_path}: {e}")
        return None

    # Generate speech
    max_frames = max_mel_frames_override if max_mel_frames_override is not None else inference_config.LOGGING_MAX_MEL_FRAMES
    
    waveform_np = generate_speech_mel(
        model=inference_model, 
        text_prompt=text_to_synthesize, 
        reference_audio_waveform=ref_waveform, 
        reference_audio_sample_rate=inference_config.TARGET_SAMPLE_RATE,
        device=used_device, 
        generation_params=custom_gen_params,
        max_mel_frames=max_frames
    )
    
    if waveform_np is not None and waveform_np.size > 0:
        output_file_path = os.path.join(output_dir, f"{output_filename_base}.wav")
        try:
            sf.write(output_file_path, waveform_np, inference_config.TARGET_SAMPLE_RATE)
            print(f"Generated audio saved to: {output_file_path}")
        except Exception as e_sf:
            print(f"Error saving audio to {output_file_path}: {e_sf}")
        return waveform_np 
    else:
        print("Speech generation failed or produced empty audio.")
        return None


def perform_varied_inference(
    model: Any, # Should be GPT2TTSMelPredictor instance
    device: torch.device,
    checkpoint_path: str, 
    texts_to_synthesize: List[str],
    reference_audio_waveform: torch.Tensor, 
    reference_audio_sample_rate: int,       
    sampling_strategies: Dict[str, Optional[Dict]], 
    target_sample_rate: int, 
    output_dir: Optional[str] = None 
):
    """
    Performs inference with a loaded model using various generation parameters.
    (Docstring from previous version, remains largely applicable)
    """
    print(f"\n--- Performing Varied Inference using checkpoint: {checkpoint_path} ---")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generated audio samples will be saved to: {os.path.abspath(output_dir)}")

    for i, text_prompt in enumerate(texts_to_synthesize):
        print(f"\n--- Synthesizing for text ({i+1}/{len(texts_to_synthesize)}): '{text_prompt}' ---")
        for strategy_name, strategy_params_dict in sampling_strategies.items():
            print(f"\nStrategy: {strategy_name}")
            
            current_gen_params_for_util = strategy_params_dict.copy() if strategy_params_dict is not None else {}
            
            if current_gen_params_for_util: 
                print(f"Params for generate_speech_mel: {current_gen_params_for_util}")
            
            max_frames_for_this_strategy = current_gen_params_for_util.pop('max_mel_frames', inference_config.LOGGING_MAX_MEL_FRAMES)

            waveform_np = generate_speech_mel(
                model=model,
                text_prompt=text_prompt,
                reference_audio_waveform=reference_audio_waveform, 
                reference_audio_sample_rate=reference_audio_sample_rate,
                device=device,
                generation_params=current_gen_params_for_util, 
                max_mel_frames=max_frames_for_this_strategy
            )
            
            if waveform_np is not None and waveform_np.size > 0:
                print(f"Generated waveform shape for '{strategy_name}': {waveform_np.shape}")
                if output_dir:
                    filename_safe_text = "".join(c if c.isalnum() else "_" for c in text_prompt[:30])
                    filename = os.path.join(output_dir, f"sample_{i+1}_{filename_safe_text}_{strategy_name.replace(' ','_')}.wav")
                    try:
                        sf.write(filename, waveform_np, target_sample_rate)
                        print(f"Saved: {filename}")
                    except Exception as e_sf:
                        print(f"Error saving {filename}: {e_sf}")
                else:
                    print("Output directory not provided. Audio not saved.")
            else:
                print(f"Speech generation failed or produced empty audio for strategy: {strategy_name}")

# --- Metric Functions (remain unchanged as they operate on waveform/text) ---

def calculate_cer(hypothesis_text: str, reference_text: str) -> float:
    try:
        hypothesis_clean = hypothesis_text.lower() 
        reference_clean = reference_text.lower()
        punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        hypothesis_clean = "".join(char for char in hypothesis_clean if char not in punctuation)
        reference_clean = "".join(char for char in reference_clean if char not in punctuation)
        cer_score = jiwer.cer(reference_clean, hypothesis_clean)
        print(f"CER: Hyp: '{hypothesis_text[:50]}...' Ref: '{reference_text[:50]}...' -> Score: {cer_score:.4f}")
        return cer_score
    except Exception as e:
        print(f"Error calculating CER: {e}. Ensure 'jiwer' is installed.")
        return 1.0 

def calculate_utmos(waveform_np: np.ndarray, sample_rate: int) -> float:
    print(f"[Placeholder UTMOS] Waveform shape: {waveform_np.shape}, SR: {sample_rate}")
    print("  Actual UTMOS calculation requires a pre-trained UTMOS model and library.")
    return 3.5 

def calculate_secs(
    generated_waveform_np: np.ndarray, 
    reference_waveform_np: np.ndarray, 
    sample_rate: int
) -> float:
    print(f"[Placeholder SECS] Gen wave: {generated_waveform_np.shape}, Ref wave: {reference_waveform_np.shape}, SR: {sample_rate}")
    print("  Actual SECS calculation requires a pre-trained Speaker Encoder model (e.g., from xTTS or SpeechBrain).")
    return 0.85 

if __name__ == '__main__':
    print("--- Running evaluation.py direct tests ---")
    hyp_test = "this is a test sentence for cer"
    ref_test = "this is a test sentence for character error rate"
    cer = calculate_cer(hyp_test, ref_test)
    print(f"Test CER: {cer:.4f}")

    sr_test = inference_config.TARGET_SAMPLE_RATE 
    dummy_audio_gen = np.random.randn(sr_test * 2) 
    dummy_audio_ref = np.random.randn(sr_test * 3) 
    
    utmos_val = calculate_utmos(dummy_audio_gen, sr_test)
    print(f"Test UTMOS (placeholder): {utmos_val:.4f}")
    
    secs_val = calculate_secs(dummy_audio_gen, dummy_audio_ref, sr_test)
    print(f"Test SECS (placeholder): {secs_val:.4f}")

    print("\nSkipping single_inference_mel and perform_varied_inference direct tests in evaluation.py as they require a trained checkpoint and reference audio.")
