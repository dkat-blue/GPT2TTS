# src/model.py
"""
Defines the GPT2TTSMelPredictor model architecture, which predicts mel spectrograms
from text, conditioned on reference audio. Uses a pre-trained HiFi-GAN vocoder
for waveform synthesis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from typing import Optional, Tuple, Union, List, Dict
import math
import os

import config as model_config

from speechbrain.inference.classifiers import EncoderClassifier 
from speechbrain.inference.vocoders import HIFIGAN 
import torchaudio.transforms as T


class SinusoidalPositionalEmbedding(nn.Module):
    """ Sinusoidal Positional Embedding for the Conditioning Encoder. """
    def __init__(self, d_model: int, max_len: int = 1000): # Max length for typical reference mel spectrograms
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len_mel_ref, cond_encoder_embed_dim)
        return self.pe[:, :x.size(1)]


class ConditioningEncoder(nn.Module):
    """
    Encodes reference audio (mel spectrogram) into a sequence of embeddings
    using Transformer Encoder layers.
    """
    def __init__(self, input_dim: int, embed_dim: int, output_dim: int, 
                 n_layers: int, n_heads: int, dropout: float = 0.1, 
                 max_ref_mel_len: int = 1000):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = SinusoidalPositionalEmbedding(embed_dim, max_len=max_ref_mel_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation=F.gelu
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(embed_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, mel_spectrogram: torch.Tensor, 
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # mel_spectrogram shape: (batch_size, n_mels, seq_len_mel_ref)
        # Permute to (batch_size, seq_len_mel_ref, n_mels) for linear layer and transformer
        x = mel_spectrogram.permute(0, 2, 1) 
        x = self.input_projection(x)
        x = x + self.pos_encoder(x) 
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = self.output_projection(x)
        x = self.layer_norm(x)
        return x # Shape: (batch_size, seq_len_mel_ref, cond_encoder_output_dim)


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler to distill a variable-length sequence (from ConditioningEncoder)
    into a fixed number of latent vectors using cross-attention.
    """
    def __init__(self, input_dim: int, latent_dim: int, num_latents: int, 
                 num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.num_latents = num_latents
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, 
                                  dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim, latent_dim * 4), nn.GELU(),
                          nn.Linear(latent_dim * 4, latent_dim), nn.Dropout(dropout))
            for _ in range(num_layers)
        ])
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(num_layers)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(num_layers)])

    def forward(self, context: torch.Tensor, 
                context_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # context shape: (batch_size, seq_len_context, input_dim)
        batch_size = context.size(0)
        latents = self.latent_queries.expand(batch_size, -1, -1) # (B, num_latents, latent_dim)

        for i in range(len(self.cross_attention_layers)):
            attn_output, _ = self.cross_attention_layers[i](
                query=latents, key=context, value=context, 
                key_padding_mask=context_padding_mask
            )
            latents = self.norm1_layers[i](latents + attn_output) # Add & Norm
            
            ffn_output = self.ffn_layers[i](latents)
            latents = self.norm2_layers[i](latents + ffn_output) # Add & Norm
            
        return latents # Shape: (batch_size, num_latents, latent_dim)


class GPT2TTSMelPredictor(nn.Module):
    """
    GPT-2 based model for Text-to-Speech that predicts mel spectrograms.
    It's conditioned on text and reference audio style.
    Uses a pre-trained HiFi-GAN vocoder for waveform synthesis.
    """
    def __init__(self, tokenizer: GPT2Tokenizer):
        super().__init__()
        self.tokenizer = tokenizer # For text tokenization
        self.gpt2_config = GPT2Config.from_pretrained(
            model_config.GPT2_MODEL_NAME,
            vocab_size=len(tokenizer), 
            n_positions=model_config.GPT2_N_POSITIONS, 
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_config.GPT2_MODEL_NAME, config=self.gpt2_config)
        # Replace GPT-2's lm_head to output mel bins instead of vocabulary logits
        self.gpt2.lm_head = nn.Linear(self.gpt2_config.hidden_size, model_config.GPT2_OUTPUT_MEL_BINS)
        print(f"Replaced GPT-2 lm_head to output {model_config.GPT2_OUTPUT_MEL_BINS} mel bins.")
        print(f"GPT-2 configured with n_positions: {self.gpt2_config.n_positions}")

        # Mel spectrogram transformation for reference audio conditioning
        self.mel_spectrogram_transform_ref = T.MelSpectrogram(
            sample_rate=model_config.TARGET_SAMPLE_RATE,
            n_fft=model_config.MEL_N_FFT, hop_length=model_config.MEL_HOP_LENGTH,
            win_length=model_config.MEL_WIN_LENGTH, n_mels=model_config.MEL_N_MELS, 
            f_min=model_config.MEL_FMIN, f_max=model_config.MEL_FMAX,
            power=2.0 # Using power spectrogram for conditioning encoder
        )

        # Conditioning components
        max_ref_mel_len = int((model_config.REFERENCE_AUDIO_MAX_DURATION_SEC * model_config.TARGET_SAMPLE_RATE) / model_config.MEL_HOP_LENGTH) + 10 
        self.conditioning_encoder = ConditioningEncoder(
            input_dim=model_config.COND_ENC_INPUT_DIM, 
            embed_dim=model_config.COND_ENC_EMBED_DIM,
            output_dim=model_config.COND_ENC_OUTPUT_DIM,
            n_layers=model_config.COND_ENC_N_ATTN_LAYERS,
            n_heads=model_config.COND_ENC_N_HEADS,
            max_ref_mel_len=max_ref_mel_len
        )
        self.perceiver_resampler = PerceiverResampler(
            input_dim=model_config.COND_ENC_OUTPUT_DIM,
            latent_dim=model_config.PERCEIVER_LATENT_DIM,
            num_latents=model_config.PERCEIVER_N_LATENTS,
            num_heads=model_config.PERCEIVER_N_HEADS,
            num_layers=model_config.PERCEIVER_N_CROSS_ATTN_LAYERS
        )

        # Speaker Encoder for SCL Proxy
        try:
            self.speaker_encoder = EncoderClassifier.from_hparams(
                source=model_config.SPEAKER_ENCODER_MODEL_NAME,
                savedir=os.path.join(model_config.PROJECT_ROOT, "pretrained_models", model_config.SPEAKER_ENCODER_MODEL_NAME.replace("/", "_")),
                run_opts={"device": model_config.DEVICE}
            )
            self.speaker_encoder.eval() # Set to evaluation mode
            print(f"Speaker encoder {model_config.SPEAKER_ENCODER_MODEL_NAME} loaded.")
        except Exception as e:
            print(f"Error loading speaker encoder: {e}. SCL Proxy will not be available.")
            self.speaker_encoder = None
        
        # Projection layer for SCL Proxy if dimensions mismatch
        if model_config.COND_ENC_OUTPUT_DIM != model_config.SCL_PROXY_PROJECTION_DIM:
            self.scl_style_projection = nn.Linear(model_config.COND_ENC_OUTPUT_DIM, model_config.SCL_PROXY_PROJECTION_DIM)
            print(f"Added SCL projection layer: {model_config.COND_ENC_OUTPUT_DIM} -> {model_config.SCL_PROXY_PROJECTION_DIM}")
        else:
            self.scl_style_projection = nn.Identity()
        
        # HiFi-GAN Vocoder
        try:
            self.hifi_gan = HIFIGAN.from_hparams(
                source=model_config.HIFIGAN_MODEL_SOURCE,
                savedir=model_config.HIFIGAN_SAVEDIR,
                run_opts={"device": model_config.DEVICE}
            )
            self.hifi_gan.eval() # Set to evaluation mode
            print(f"HiFi-GAN vocoder {model_config.HIFIGAN_MODEL_SOURCE} loaded.")
        except Exception as e:
            print(f"Error loading HiFi-GAN vocoder: {e}. Waveform generation will not be available.")
            self.hifi_gan = None
            
        # Input projection for mel frames fed into GPT-2 during teacher forcing/generation
        self.mel_input_projection = nn.Linear(model_config.MEL_N_MELS, self.gpt2_config.hidden_size)
        # Learnable start-of-mel-sequence embedding
        self.start_mel_embedding = nn.Parameter(torch.randn(1, 1, self.gpt2_config.hidden_size))


    def get_speaker_embedding(self, audio_waveform: torch.Tensor, sample_rate: int) -> Optional[torch.Tensor]:
        """ Extracts speaker embedding from an audio waveform. """
        if self.speaker_encoder is None: return None
        if audio_waveform.ndim == 1: audio_waveform = audio_waveform.unsqueeze(0) # Add batch dim
        
        # Resample to 16kHz if necessary, as expected by ECAPA-TDNN
        if sample_rate != 16000: 
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000).to(audio_waveform.device)
            audio_waveform = resampler(audio_waveform)
        try:
            with torch.no_grad():
                # SpeechBrain's ECAPA-TDNN encode_batch returns (batch, 1, embed_dim)
                embeddings = self.speaker_encoder.encode_batch(audio_waveform).squeeze(dim=1) 
            return embeddings # Shape: (batch, embed_dim)
        except Exception as e:
            print(f"Error during speaker embedding extraction: {e}"); return None

    def forward(
        self,
        text_input_ids: torch.Tensor,                     
        reference_audio_waveform: torch.Tensor,           
        reference_audio_sample_rate: int,
        target_mel_spectrogram: torch.Tensor, # For loss calculation (B, N_MELS, S_mel_target)            
        input_mel_spectrogram_shifted: torch.Tensor, # For teacher forcing (B, N_MELS, S_mel_input)     
        text_attention_mask: Optional[torch.Tensor] = None, 
        mel_actual_frame_mask: Optional[torch.Tensor] = None # (B, S_mel_target) True for actual frames
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the GPT2TTSMelPredictor model.
        Predicts mel spectrograms based on text and reference audio style.
        """
        batch_size = text_input_ids.size(0)
        device = text_input_ids.device

        # 1. Process Reference Audio for Style Conditioning
        reference_audio_waveform = reference_audio_waveform.to(device)
        ref_mel_spec_cond = self.mel_spectrogram_transform_ref(reference_audio_waveform) 
        cond_enc_output = self.conditioning_encoder(ref_mel_spec_cond) 
        style_embeddings = self.perceiver_resampler(cond_enc_output) # (B, num_latents, gpt_hidden_size)  

        # 2. Prepare inputs for GPT-2's transformer
        text_embeds = self.gpt2.transformer.wte(text_input_ids) # (B, S_text, gpt_hidden_size)
        
        # Project input mel frames (teacher-forcing) to GPT-2's hidden size
        # input_mel_spectrogram_shifted shape: (B, N_MELS, S_mel_input)
        input_mel_embeds_unprojected = input_mel_spectrogram_shifted.permute(0, 2, 1) # (B, S_mel_input, N_MELS)
        input_mel_embeds = self.mel_input_projection(input_mel_embeds_unprojected) # (B, S_mel_input, gpt_hidden_size)

        # Prepend start-of-mel embedding to the input mel sequence for GPT-2
        start_mel_emb_batch = self.start_mel_embedding.expand(batch_size, -1, -1) # (B, 1, gpt_hidden_size)
        
        # Teacher-forcing input for mel part: [START_MEL_EMB, mel_frame_0_embed, ..., mel_frame_T-2_embed]
        # This sequence will be used to predict [mel_frame_0, ..., mel_frame_T-1]
        if input_mel_embeds.size(1) > 0: 
            # Use (T-1) frames from input_mel_embeds, as the last one is not used to predict a next one
            gpt2_mel_input_sequence = torch.cat([start_mel_emb_batch, input_mel_embeds[:, :-1, :]], dim=1) 
        else: 
            # If input_mel_embeds is empty (e.g. target is 1 frame long), only use start_mel_embedding
            gpt2_mel_input_sequence = start_mel_emb_batch
        
        # Concatenate all embeddings for GPT-2 input
        inputs_embeds = torch.cat([style_embeddings, text_embeds, gpt2_mel_input_sequence], dim=1)

        # 3. Create attention mask for the combined sequence
        if text_attention_mask is None: text_attention_mask = torch.ones_like(text_input_ids)
        style_attention_mask = torch.ones(style_embeddings.shape[:2], dtype=torch.long, device=device)
        mel_input_attention_mask = torch.ones(gpt2_mel_input_sequence.shape[:2], dtype=torch.long, device=device)
        # Note: If gpt2_mel_input_sequence could have padding, its mask would need to reflect that.
        
        attention_mask = torch.cat([style_attention_mask, text_attention_mask, mel_input_attention_mask], dim=1)
        
        # Safeguard against exceeding GPT-2's max positions
        current_seq_len = inputs_embeds.size(1)
        if current_seq_len > self.gpt2_config.n_positions:
            # This should be prevented by dataset preparation (MAX_TEXT_LEN_DATASET, MAX_MEL_FRAMES_DATASET)
            print(f"Warning: Input sequence length ({current_seq_len}) exceeds GPT-2 max positions ({self.gpt2_config.n_positions}). Truncating.")
            inputs_embeds = inputs_embeds[:, :self.gpt2_config.n_positions, :]
            attention_mask = attention_mask[:, :self.gpt2_config.n_positions]


        # 4. Pass through GPT-2 transformer blocks
        transformer_outputs = self.gpt2.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = transformer_outputs.last_hidden_state 
        
        # Extract hidden states corresponding to the mel prediction part
        num_style_latents = style_embeddings.size(1)
        num_text_tokens = text_embeds.size(1)
        start_mel_pred_idx = num_style_latents + num_text_tokens
        
        # The number of mel frames to predict is target_mel_spectrogram.size(2)
        # The gpt2_mel_input_sequence has length equal to the number of frames we predict.
        num_mel_frames_to_predict = gpt2_mel_input_sequence.size(1)
        
        # Hidden states for mel prediction
        mel_hidden_states = hidden_states[:, start_mel_pred_idx : start_mel_pred_idx + num_mel_frames_to_predict, :]
        # Predict mel frames using the modified lm_head
        predicted_mel_frames_flat = self.gpt2.lm_head(mel_hidden_states) # (B, num_mel_frames_to_predict, N_MELS)
        # Reshape to (B, N_MELS, num_mel_frames_to_predict)
        predicted_mel_spectrogram = predicted_mel_frames_flat.permute(0, 2, 1)


        # 5. Calculate Mel Prediction Loss
        # Compare with target_mel_spectrogram
        len_predicted = predicted_mel_spectrogram.size(2)
        len_target = target_mel_spectrogram.size(2)
        # The lengths should match if teacher forcing input was prepared correctly based on target length
        min_len = min(len_predicted, len_target) 
        
        mel_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        if min_len > 0 : 
            # Use mel_actual_frame_mask to compute loss only on actual (unpadded) frames
            # Mask shape: (B, S_mel_target), needs to be (B, 1, S_mel_target) for broadcasting
            current_mask = mel_actual_frame_mask[:, :min_len].unsqueeze(1).expand_as(predicted_mel_spectrogram[:,:,:min_len])

            if model_config.MEL_LOSS_TYPE == "L1":
                loss_per_frame = F.l1_loss(
                    predicted_mel_spectrogram[:, :, :min_len], 
                    target_mel_spectrogram[:, :, :min_len], 
                    reduction='none' # Get per-element loss
                )
            elif model_config.MEL_LOSS_TYPE == "MSE":
                loss_per_frame = F.mse_loss(
                    predicted_mel_spectrogram[:, :, :min_len], 
                    target_mel_spectrogram[:, :, :min_len],
                    reduction='none' # Get per-element loss
                )
            else:
                raise ValueError(f"Unknown mel_loss_type: {model_config.MEL_LOSS_TYPE}")
            
            masked_loss = loss_per_frame * current_mask # Apply mask
            # Average loss over actual frames only
            mel_loss = masked_loss.sum() / current_mask.sum().clamp(min=1e-8) 

        if not mel_loss.requires_grad and min_len == 0 : 
             mel_loss = mel_loss.clone().requires_grad_(True) # Ensure grad if loss is 0 due to no frames


        # 6. Calculate SCL Proxy Loss
        scl_proxy_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        ref_spk_embed = None 
        if self.speaker_encoder and model_config.SCL_PROXY_WEIGHT > 0:
            ref_spk_embed = self.get_speaker_embedding(reference_audio_waveform, reference_audio_sample_rate)
            
            if ref_spk_embed is not None:
                pooled_style_embed = style_embeddings.mean(dim=1) # (B, perceiver_latent_dim)
                projected_style_embed = self.scl_style_projection(pooled_style_embed) 
                
                if projected_style_embed.shape == ref_spk_embed.shape:
                    if model_config.SCL_PROXY_LOSS_TYPE == "cosine":
                        scl_proxy_loss = 1.0 - F.cosine_similarity(projected_style_embed, ref_spk_embed, dim=1).mean()
                    elif model_config.SCL_PROXY_LOSS_TYPE == "mse":
                        scl_proxy_loss = F.mse_loss(projected_style_embed, ref_spk_embed)
                else:
                    # Warning only if dimensions mismatch, not tied to global_step
                    print(f"Warning (SCL Proxy dim mismatch): Projected Style: {projected_style_embed.shape}, Ref Spk: {ref_spk_embed.shape}")


        return {
            "mel_loss": mel_loss,
            "scl_proxy_loss": scl_proxy_loss,
            "predicted_mel_spectrogram": predicted_mel_spectrogram, 
            "style_embeddings": style_embeddings, 
            "reference_speaker_embedding": ref_spk_embed 
        }

    def decode_mels_to_waveform(self, mel_spectrograms: torch.Tensor) -> Optional[torch.Tensor]:
        """ Decodes mel spectrograms to waveform using the HiFi-GAN vocoder. """
        if self.hifi_gan is None:
            print("HiFi-GAN vocoder not available.")
            return None
        
        # HiFi-GAN's first Conv1d expects input (B, N_MELS, T_FRAMES).
        # `mel_spectrograms` should already be in this format.
        with torch.no_grad():
            waveforms = self.hifi_gan.decode_batch(mel_spectrograms) 
        return waveforms.squeeze(1) # Output shape: (B, T_samples)

    @classmethod
    def from_pretrained_custom(cls, tokenizer: GPT2Tokenizer, 
                               checkpoint_path: Optional[str] = None, 
                               device: torch.device = model_config.DEVICE):
        """ Loads model from checkpoint or initializes new. """
        model = cls(tokenizer=tokenizer) 
        if checkpoint_path:
            try:
                state_dict = torch.load(checkpoint_path, map_location=device)
                if all(k.startswith('module.') for k in state_dict.keys()): # Handle DataParallel prefix
                    state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
                
                current_model_dict = model.state_dict()
                filtered_state_dict = {}
                for k, v in state_dict.items():
                    if k in current_model_dict and current_model_dict[k].size() == v.size():
                        # Exclude speaker_encoder and hifi_gan from checkpoint loading
                        if not k.startswith("speaker_encoder.") and not k.startswith("hifi_gan."):
                            filtered_state_dict[k] = v
                model.load_state_dict(filtered_state_dict, strict=False)
                print(f"Loaded model weights from {checkpoint_path} (excluding speaker_encoder and hifi_gan).")
            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_path}: {e}. Initializing new model.")
        model.to(device)
        return model

    # Methods for Hugging Face `generate` compatibility (if used, needs careful adaptation for mel)
    def get_input_embeddings(self): 
        return self.gpt2.transformer.wte

    def get_output_embeddings(self): 
        return self.gpt2.lm_head # This is now the mel prediction linear layer

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        Prepares inputs for generation.
        This is primarily for Hugging Face's `model.generate()` method.
        For manual autoregressive loops (like in utils.generate_speech_mel),
        input preparation is handled directly in the loop.
        """
        # `input_ids` in this context (if using HF generate for mel) would represent
        # the embedding of the previously generated mel frame or a start token.
        # This function is a STUB and would need significant work if relying on model.generate().
        current_input_embeds = input_ids 

        attention_mask = kwargs.get("attention_mask")
        if past_key_values is not None and attention_mask is not None:
            new_mask_token = torch.ones((attention_mask.size(0), 1), dtype=torch.long, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, new_mask_token], dim=1)

        if current_input_embeds.ndim == 2: # Ensure (B, 1, H) for single token embedding
            current_input_embeds = current_input_embeds.unsqueeze(1)
            
        return {
            "inputs_embeds": current_input_embeds, 
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        }
