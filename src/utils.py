# src/utils.py
"""
Utilities for GPT2TTS: speech generation, Encodec diagnostics, audio logging.
"""
import torch
from IPython.display import Audio as IPyAudio, display 
from typing import Optional, List, Dict
from collections import Counter 
import os
import soundfile as sf 
import numpy as np
from transformers import GenerationConfig
from torch.utils.data import Dataset 

import config 

def generate_speech(
    model, text_prompt: str, device: torch.device, 
    generation_params: Optional[Dict] = None
) -> Optional[np.ndarray]:
    """
    Generates speech waveform from text using the GPT2TTS model.

    Args:
        model: Initialized GPT2TTS model instance.
        text_prompt: Text to synthesize.
        device: Torch device for generation.
        generation_params: Optional dict for Hugging Face .generate().
                           Defaults include anti-mode-collapse settings.

    Returns:
        NumPy array of the waveform, or None on failure.
    """
    model.eval() 

    if generation_params is None:
        # Default generation parameters
        temp_tokenizer_output = model.tokenizer(text_prompt, return_tensors="pt", truncation=True, max_length=model.config.n_positions // 3)
        prompt_len_calc = temp_tokenizer_output.input_ids.shape[1] + 1 
        max_gen_len_model = model.config.n_positions - prompt_len_calc - 1 
        effective_max_new_tokens = max(100, min(config.LOGGING_MAX_NEW_TOKENS, max_gen_len_model if max_gen_len_model > 0 else 100))
        generation_params = {
            "max_new_tokens": effective_max_new_tokens, "temperature": 0.75, "top_p": 0.9, 
            "do_sample": True, "repetition_penalty": 1.0, # 1.0 to disable problematic processor
        }
    
    try:
        initial_input_ids, initial_attention_mask, text_token_len = \
            model.prepare_prompt_for_generation([text_prompt], device=device)
    except ValueError as e:
        print(f"Error preparing prompt for '{text_prompt[:30]}...': {e}")
        return None
    
    num_prompt_tokens = initial_input_ids.shape[1]
    requested_max_new = generation_params.get("max_new_tokens", model.config.n_positions - num_prompt_tokens - 1)
    adjusted_max_new = min(requested_max_new, model.config.n_positions - num_prompt_tokens - 1)

    if adjusted_max_new <= 0:
        print(f"Warning: Prompt too long or max_new_tokens too small. Effective max_new_tokens: {adjusted_max_new}. Cannot generate.")
        return None

    current_gen_params = generation_params.copy() 
    current_gen_params["max_new_tokens"] = adjusted_max_new
    
    gen_config_hf = GenerationConfig( 
        pad_token_id=model.eos_audio_token_id, eos_token_id=model.eos_audio_token_id,
        bos_token_id=model.bos_audio_token_id, 
        suppress_tokens=[model.bos_audio_token_id], **current_gen_params
    )

    log_params_str = (f"max_new_tokens={adjusted_max_new}, temp={gen_config_hf.temperature:.2f}, "
                      f"top_p={gen_config_hf.top_p:.2f}, suppress_tokens={[model.bos_audio_token_id]}")
    if gen_config_hf.repetition_penalty != 1.0: log_params_str += f", repetition_penalty={gen_config_hf.repetition_penalty:.2f}"
    print(f"Generating with: {log_params_str}")
          
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=initial_input_ids, attention_mask=initial_attention_mask, 
            generation_config=gen_config_hf, prompt_text_length=text_token_len 
        )
    
    gen_audio_tokens_full = output_sequences[0] 
    gen_audio_tokens_eos = gen_audio_tokens_full[num_prompt_tokens:] if len(gen_audio_tokens_full) > num_prompt_tokens else torch.tensor([], dtype=torch.long, device=device)

    # --- Token Inspection ---
    # print(f"\n--- Generated Token Inspection for: '{text_prompt[:30]}...' ---")
    # print(f"Raw generated audio tokens (after prompt, first 50): {gen_audio_tokens_eos[:50].tolist()}")
    # print(f"Number of raw generated audio tokens: {len(gen_audio_tokens_eos)}")

    gen_audio_tokens_clean = gen_audio_tokens_eos[:-1] if len(gen_audio_tokens_eos) > 0 and gen_audio_tokens_eos[-1] == model.eos_audio_token_id else gen_audio_tokens_eos
    
    # print(f"Cleaned audio tokens (first 50): {gen_audio_tokens_clean[:50].tolist()}")
    # print(f"Number of cleaned audio tokens: {len(gen_audio_tokens_clean)}")
    # if len(gen_audio_tokens_clean) > 0:
    #     token_counts = Counter(gen_audio_tokens_clean.cpu().numpy())
    #     display_token_counts = {int(k): int(v) for k, v in token_counts.most_common(10)}
    #     print(f"Most common generated tokens: {display_token_counts}")
    #     print(f"Number of unique tokens generated: {len(token_counts)}")
    # print(f"--- End Token Inspection ---\n")

    if len(gen_audio_tokens_clean) == 0:
        print("No audio tokens generated after prompt and EOS stripping.")
        return None
    
    # Filter special tokens before Encodec decoding
    valid_codebook_tokens = gen_audio_tokens_clean[gen_audio_tokens_clean < model.codebook_size]
    if len(valid_codebook_tokens) != len(gen_audio_tokens_clean):
        # print(f"Warning: Special tokens filtered. Original: {len(gen_audio_tokens_clean)}, Filtered: {len(valid_codebook_tokens)}") # Optional
        if len(valid_codebook_tokens) == 0:
            print("No valid codebook tokens remain after filtering.")
            return None
            
    if len(valid_codebook_tokens) == 0: 
        print("No valid codebook tokens to decode.")
        return None

    waveform = model.decode_audio_tokens(valid_codebook_tokens.to(model.device)) 
    return waveform.cpu().numpy()


def run_encodec_path_verification(model, dataset: Dataset, device: torch.device):
    """
    Performs Encodec encode/decode diagnostic on a dataset sample.
    Compares original audio with reconstructed audio.
    """
    if not dataset or len(dataset) == 0:
        print("Dataset unavailable/empty for Encodec verification.")
        return

    print("\n--- Running Encodec Path Verification ---")
    try:
        sample_data = dataset[0] 
        waveform_np, text = sample_data['audio_waveform'], sample_data['text']
        print(f"Using GT audio for text: '{text[:100]}...' ")

        inputs = model.processor(raw_audio=waveform_np, sampling_rate=model.processor.sampling_rate, return_tensors="pt")
        input_values = inputs["input_values"].to(device)
        padding_mask = inputs.get("padding_mask", torch.ones_like(input_values, dtype=torch.bool)).to(device)


        with torch.no_grad():
            model.codec.to(device)
            encoder_outputs = model.codec.encode(input_values, padding_mask, bandwidth=config.ENCODEC_BANDWIDTH_TRAIN)
        
        gt_audio_codes = encoder_outputs.audio_codes # (1, B, Nq, T_frames)
        # print(f"GT audio codes shape: {gt_audio_codes.shape}")
        
        assert gt_audio_codes.dim() == 4 and gt_audio_codes.shape[0] == 1 and \
               gt_audio_codes.shape[2] == model.num_audio_codebooks, "Unexpected gt_audio_codes shape."

        scales = torch.ones((gt_audio_codes.size(0), gt_audio_codes.size(1)), device=device, dtype=torch.float32)

        with torch.no_grad():
            decoded_output = model.codec.decode(gt_audio_codes, scales) # No padding_mask needed for decode
        
        decoded_waveform_np = decoded_output[0].squeeze(0).squeeze(0).cpu().numpy()
        # print(f"Decoded waveform shape (diagnostic): {decoded_waveform_np.shape}")

        print("Playing RECONSTRUCTED GT audio (encode & decode):")
        display(IPyAudio(decoded_waveform_np, rate=config.TARGET_SAMPLE_RATE))
        print("Playing ORIGINAL GT audio:")
        display(IPyAudio(waveform_np.mean(axis=0) if waveform_np.ndim > 1 else waveform_np, rate=config.TARGET_SAMPLE_RATE))

    except Exception as e:
        print(f"Error during Encodec verification: {e}")
        import traceback; traceback.print_exc()
    finally:
        print("--- Encodec Path Verification Done ---\n")

def log_audio_samples(
    model, epoch: int, device: torch.device, 
    sample_texts: List[str] = config.LOGGING_SAMPLE_TEXTS, 
    audio_logging_dir: str = config.AUDIO_LOGGING_DIR,
    target_sample_rate: int = config.TARGET_SAMPLE_RATE
):
    """Generates and saves audio samples for predefined texts during training."""
    if not sample_texts: return
    
    print(f"\n--- Logging audio samples for epoch {epoch} ---")
    model.eval()
    with torch.no_grad():
        for sample_idx, text_prompt in enumerate(sample_texts):
            print(f"Generating for: '{text_prompt}'")
            # Uses default generation_params from generate_speech for consistency
            gen_waveform = generate_speech(model, text_prompt, device, generation_params=None) 
            
            if gen_waveform is not None and gen_waveform.size > 0:
                filename = os.path.join(audio_logging_dir, f"epoch_{epoch:03d}_sample_{sample_idx+1}.wav")
                try:
                    sf.write(filename, gen_waveform, target_sample_rate)
                    print(f"Saved: {filename}")
                except Exception as e_sf:
                    print(f"Error saving {filename}: {e_sf}")
            else:
                print(f"Failed to generate for logging: '{text_prompt}'")
    model.train()
    print(f"--- End audio logging for epoch {epoch} ---\n")
