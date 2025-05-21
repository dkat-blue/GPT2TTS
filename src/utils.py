# src/utils.py
"""
Utility functions for the TTS model, primarily for speech generation
using the GPT2TTSMelPredictor and its integrated HiFi-GAN vocoder.
"""
import torch
from typing import Optional, Dict, Any
import numpy as np

import config as utils_config
# from model import GPT2TTSMelPredictor # Using 'Any' for type hint to avoid circular import issues

@torch.no_grad() # Disable gradient calculations for inference
def generate_speech_mel(
    model: Any, # Expected: GPT2TTSMelPredictor instance
    text_prompt: str,
    reference_audio_waveform: torch.Tensor, # Raw waveform tensor (num_ref_samples,)
    reference_audio_sample_rate: int,
    device: torch.device,
    generation_params: Optional[Dict[str, Any]] = None, # For future parameters like temperature
    max_mel_frames: int = utils_config.MAX_MEL_FRAMES_TO_GENERATE
) -> Optional[np.ndarray]:
    """
    Generates speech waveform from text using the GPT2TTSMelPredictor model.
    The process involves:
    1. Tokenizing text and preparing reference audio for conditioning.
    2. Autoregressively predicting mel spectrogram frames using GPT-2.
    3. Vocoding the predicted mel spectrogram using the model's HiFi-GAN.
    """
    model.eval() # Ensure model is in evaluation mode
    gpt2_tokenizer = model.tokenizer 

    # 1. Tokenize text prompt
    # Use a reasonable truncation length for the text part based on GPT-2's context
    text_max_len_for_prompt = model.gpt2_config.n_positions // 4 # Heuristic, can be tuned
    text_tokens_dict = gpt2_tokenizer(
        text_prompt,
        max_length=text_max_len_for_prompt, 
        padding=False, # No padding for single prompt generation
        truncation=True,
        return_tensors="pt"
    )
    text_input_ids = text_tokens_dict.input_ids.to(device) # Shape: (1, seq_len_text)

    # 2. Prepare reference audio for style conditioning
    if reference_audio_waveform.ndim == 1:
        reference_audio_waveform = reference_audio_waveform.unsqueeze(0) # Add batch dim: (1, num_ref_samples)
    reference_audio_waveform = reference_audio_waveform.to(device)
    
    # Transform reference audio to mel spectrogram for the conditioning encoder
    ref_mel_spec_cond = model.mel_spectrogram_transform_ref(reference_audio_waveform)
    # Get style embeddings via ConditioningEncoder and PerceiverResampler
    cond_enc_output = model.conditioning_encoder(ref_mel_spec_cond)
    style_embeddings = model.perceiver_resampler(cond_enc_output) # Shape: (1, num_latents, gpt_hidden_size)

    # 3. Autoregressive Mel Spectrogram Generation
    text_embeds = model.gpt2.transformer.wte(text_input_ids) # Shape: (1, seq_len_text, gpt_hidden_size)
    
    # Initial input for mel generation is the learned start_mel_embedding
    # Shape: (batch_size=1, 1, gpt_hidden_size)
    current_mel_input_embed = model.start_mel_embedding.expand(text_input_ids.size(0), -1, -1) 
    
    generated_mel_frames_list = [] # To store predicted mel frames (each of shape (1, N_MELS))
    past_key_values = None # For caching past computations in GPT-2

    # Determine the full prompt embedding for the first step of mel generation
    # This includes style, text, and the start_mel_embedding
    initial_prompt_embeds = torch.cat([style_embeddings, text_embeds, current_mel_input_embed], dim=1)
    # Initial attention mask covers the full prompt
    current_attention_mask = torch.ones(initial_prompt_embeds.shape[:2], dtype=torch.long, device=device)

    for i in range(max_mel_frames):
        if i == 0: # First step: use the full initial prompt
            inputs_for_gpt_step = initial_prompt_embeds
            attn_mask_for_gpt_step = current_attention_mask
        else: # Subsequent steps: only the embedding of the last predicted mel frame
            inputs_for_gpt_step = current_mel_input_embed
            # Attention mask needs to be extended for the new token if using past_key_values
            attn_mask_for_gpt_step = torch.cat(
                [current_attention_mask, 
                 torch.ones((current_attention_mask.size(0), 1), dtype=torch.long, device=device)], 
                dim=1
            )
            current_attention_mask = attn_mask_for_gpt_step # Update for next iteration if needed by model

        # Pass through GPT-2 transformer blocks
        transformer_outputs = model.gpt2.transformer(
            inputs_embeds=inputs_for_gpt_step,
            attention_mask=attn_mask_for_gpt_step if i == 0 else None, # Full mask for first step, None if past_key_values handles it
            past_key_values=past_key_values,
            use_cache=True, # Enable caching for faster generation
            return_dict=True
        )
        # Get hidden state of the last token in the current input sequence
        hidden_states_last_token = transformer_outputs.last_hidden_state[:, -1:, :] 
        # Predict the next mel frame using the modified lm_head
        predicted_mel_frame_vector = model.gpt2.lm_head(hidden_states_last_token) # Shape: (1, 1, N_MELS)
        
        predicted_mel_frame = predicted_mel_frame_vector.squeeze(1) # Shape: (1, N_MELS)
        generated_mel_frames_list.append(predicted_mel_frame)
        
        # Prepare the predicted mel frame as input for the next step
        current_mel_input_embed = model.mel_input_projection(predicted_mel_frame) # Project to gpt_hidden_size
        current_mel_input_embed = current_mel_input_embed.unsqueeze(1) # Add sequence dim: (1, 1, gpt_hidden_size)
        
        past_key_values = transformer_outputs.past_key_values # Update past key values

        # TODO: Implement a mechanism to predict an End-Of-Sequence (EOS) for mels
        # This could involve training the model to output a special value or using a separate stop token predictor.
        # For now, generation stops at max_mel_frames.

    if not generated_mel_frames_list:
        print("Warning: No mel frames were generated.")
        return None

    # Concatenate all generated mel frames
    # Each frame in list is (1, N_MELS), cat along dim 0 -> (S_mel_gen, N_MELS)
    predicted_mel_spectrogram_flat = torch.cat(generated_mel_frames_list, dim=0) 
    # Transpose and add batch dim for vocoder: (S_mel_gen, N_MELS) -> (N_MELS, S_mel_gen) -> (1, N_MELS, S_mel_gen)
    predicted_mel_spectrogram_for_vocoder = predicted_mel_spectrogram_flat.T.unsqueeze(0)    

    # 4. Vocode mel spectrogram to waveform using the model's HiFi-GAN
    if model.hifi_gan is None:
        print("HiFi-GAN vocoder not loaded in the model. Cannot generate waveform.")
        return None
        
    # decode_mels_to_waveform expects (B, N_MELS, T_FRAMES)
    generated_waveform = model.decode_mels_to_waveform(predicted_mel_spectrogram_for_vocoder) 
    
    if generated_waveform is None:
        return None
        
    return generated_waveform.squeeze(0).cpu().numpy() # Return as NumPy array (num_samples,)

