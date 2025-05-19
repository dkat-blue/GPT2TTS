# src/model.py
"""
Defines the GPT2TTS model, combining GPT-2 for autoregressive processing
and Encodec for audio tokenization/decoding.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence 
from transformers import (
    GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel, GPT2Config,
    EncodecModel, AutoProcessor, GenerationConfig, GenerationMixin  
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Optional, Tuple, Dict
import numpy as np
import config 

class GPT2TTS(GPT2PreTrainedModel, GenerationMixin):
    """
    GPT2TTS model for text-to-speech.
    Uses GPT-2 for autoregressive generation of discrete audio tokens from Encodec.
    """
    def __init__(self, tokenizer: GPT2Tokenizer, n_special_tokens: int = 2): 
        """
        Initializes the GPT2TTS model.

        Args:
            tokenizer: Initialized GPT2Tokenizer. Its properties (vocab_size, pad_token_id)
                       are used to configure the internal GPT2Model.
            n_special_tokens: Number of special audio tokens (e.g., BOS_AUDIO, EOS_AUDIO).
        """
        base_config_name = tokenizer.name_or_path if tokenizer.name_or_path else config.GPT2_MODEL_NAME
        base_gpt2_config = GPT2Config.from_pretrained(base_config_name)

        # Construct final GPT-2 config, ensuring vocab consistency with the tokenizer.
        final_gpt2_model_config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            n_positions=base_gpt2_config.n_positions,
            n_embd=base_gpt2_config.n_embd,
            n_layer=base_gpt2_config.n_layer,
            n_head=base_gpt2_config.n_head,
            n_inner=base_gpt2_config.n_inner,
            activation_function=base_gpt2_config.activation_function,
            resid_pdrop=base_gpt2_config.resid_pdrop,
            embd_pdrop=base_gpt2_config.embd_pdrop,
            attn_pdrop=base_gpt2_config.attn_pdrop,
            layer_norm_epsilon=base_gpt2_config.layer_norm_epsilon,
            initializer_range=base_gpt2_config.initializer_range,
            tie_word_embeddings=False, # Word embeddings not tied to output layer.
            name_or_path=base_config_name 
        )
        
        super().__init__(final_gpt2_model_config) 
        self.config = final_gpt2_model_config     
        self.tokenizer = tokenizer          

        self.gpt2_model = GPT2Model(self.config) 

        self.codec = EncodecModel.from_pretrained(config.ENCODEC_MODEL_NAME)
        self.processor = AutoProcessor.from_pretrained(config.ENCODEC_MODEL_NAME)
        
        self.embedding_padding_value = 0.0 # Padding for embedding tensors.

        # Loss function with label smoothing, ignores padding index -100.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=config.LABEL_SMOOTHING) 
        
        self.n_special_tokens = n_special_tokens 
        self.num_audio_codebooks = self.codec.config.num_quantizers 
        self.codebook_size = self.codec.config.codebook_size 
        
        self.audio_vocab_size = self.codebook_size + self.n_special_tokens
        self.audio_emb = nn.Embedding(self.audio_vocab_size, self.config.n_embd) # Audio token embeddings.
        self.lm_head = nn.Linear(self.config.n_embd, self.audio_vocab_size, bias=False) # Projects to audio vocab.
        
        self.bos_audio_token_id = self.codebook_size 
        self.eos_audio_token_id = self.codebook_size + 1 
        
        print(f"GPT2TTS initialized. Audio vocab size: {self.audio_vocab_size}")
        print(f"GPT2Model using config: vocab_size={self.config.vocab_size}, pad_token_id={self.config.pad_token_id}")

    def get_input_embeddings(self) -> nn.Embedding:
        """Returns GPT-2's input (text) token embeddings."""
        return self.gpt2_model.wte 

    def get_output_embeddings(self) -> nn.Linear:
        """Returns the output LM head for audio tokens."""
        return self.lm_head 

    def can_generate(self) -> bool: 
        """Flags model for Hugging Face .generate() compatibility."""
        return True

    def _escape_fstring_text(self, text_segment: str) -> str:
        """Escapes special characters for safe f-string logging."""
        return text_segment.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r').replace("'", "\\'").replace('"', '\\"')

    def preprocess_text_and_audio(self, texts: List[str], audio_waveforms: List[np.ndarray], 
                                  device: torch.device) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Preprocesses batch of texts and audio: text tokenization, Encodec audio tokenization,
        and flattening of audio codes. Skips erroneous samples.

        Args:
            texts: List of input text strings.
            audio_waveforms: List of NumPy audio waveforms.
            device: Torch device for tensors.

        Returns:
            Tuple: (list of text token Tensors, list of flattened audio token Tensors).
        """
        valid_text_tokens_list = []
        valid_audio_tokens_list = []

        for text, waveform in zip(texts, audio_waveforms):
            try:
                max_text_tokens = self.config.n_positions // 3 
                text_encoding = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_text_tokens) 
                current_text_tokens = text_encoding.input_ids.squeeze(0).to(device) 
                if current_text_tokens.numel() == 0 or waveform.size == 0: continue

                inputs = self.processor(raw_audio=waveform, sampling_rate=self.processor.sampling_rate, return_tensors="pt")
                input_values = inputs["input_values"].to(device) 
                if input_values.numel() == 0: continue
                
                padding_mask_proc = inputs.get("padding_mask", None) 
                if padding_mask_proc is not None: padding_mask_proc = padding_mask_proc.to(device)

                with torch.no_grad(): 
                    self.codec.to(input_values.device)
                    encoder_outputs = self.codec.encode(
                        input_values, padding_mask=padding_mask_proc, 
                        bandwidth=config.ENCODEC_BANDWIDTH_TRAIN 
                    )
                codes_from_encodec = encoder_outputs.audio_codes # (1, B, Nq, T) or (1,1,Nq,T)
                
                current_codes = codes_from_encodec.squeeze(0).squeeze(0) # Expect (Nq, T) for single sample
                                
                if current_codes.dim() != 2 or current_codes.shape[0] != self.num_audio_codebooks or current_codes.shape[1] == 0:
                    continue
                                
                flattened_audio_tokens = current_codes.T.reshape(-1).to(device)
                if flattened_audio_tokens.numel() == 0: continue

                valid_text_tokens_list.append(current_text_tokens)
                valid_audio_tokens_list.append(flattened_audio_tokens)
            except Exception:
                continue # Skip problematic sample
        return valid_text_tokens_list, valid_audio_tokens_list

    def forward(
        self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, 
        audio_waveforms: Optional[List[np.ndarray]] = None, texts: Optional[List[str]] = None, 
        return_loss: bool = False, inputs_embeds: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, 
        use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, 
        prompt_text_length: Optional[int] = None, 
        current_max_audio_tokens_for_curriculum: Optional[int] = None, 
        **kwargs 
    ) -> CausalLMOutputWithPast: 
        """
        Forward pass. Handles training/validation (if return_loss=True) or inference.
        For training: constructs combined embeddings, calculates loss.
        For inference: processes inputs_embeds or input_ids for generation.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if return_loss: 
            if texts is None or audio_waveforms is None: 
                raise ValueError("texts and audio_waveforms required for return_loss=True.")
            device = next(self.parameters()).device 
            text_token_ids_list, audio_token_ids_list = self.preprocess_text_and_audio(texts, audio_waveforms, device)
            
            if not text_token_ids_list: 
                return CausalLMOutputWithPast(loss=torch.tensor(0.0, device=device, requires_grad=True), 
                                              logits=torch.empty((0,0,self.audio_vocab_size), device=device))

            batch_input_embeds_list, batch_labels_list = [], []
            for text_tokens, audio_tokens_original in zip(text_token_ids_list, audio_token_ids_list):
                num_text_tokens = text_tokens.shape[0]
                max_audio_len_context = self.config.n_positions - num_text_tokens - 2 # For BOS & EOS audio
                if max_audio_len_context <= 0: continue 

                effective_max_audio_len = min(max_audio_len_context, 
                                              current_max_audio_tokens_for_curriculum or max_audio_len_context)
                audio_tokens_tr = audio_tokens_original[:effective_max_audio_len]
                if audio_tokens_tr.numel() == 0: continue

                text_emb = self.gpt2_model.wte(text_tokens) 
                bos_emb = self.audio_emb(torch.tensor([self.bos_audio_token_id], device=device))
                audio_emb = self.audio_emb(audio_tokens_tr)
                eos_emb = self.audio_emb(torch.tensor([self.eos_audio_token_id], device=device))
                
                current_embeds = torch.cat([text_emb, bos_emb, audio_emb, eos_emb], dim=0)
                current_labels = torch.cat([
                    torch.full((num_text_tokens + 1,), -100, device=device, dtype=torch.long), # Ignore text & BOS
                    audio_tokens_tr, torch.tensor([self.eos_audio_token_id], device=device, dtype=torch.long)
                ], dim=0)
                
                if current_embeds.shape[0] != current_labels.shape[0]: continue
                batch_input_embeds_list.append(current_embeds)
                batch_labels_list.append(current_labels)

            if not batch_input_embeds_list: 
                return CausalLMOutputWithPast(loss=torch.tensor(0.0, device=device, requires_grad=True),
                                              logits=torch.empty((0,0,self.audio_vocab_size), device=device))

            padded_embeds = pad_sequence(batch_input_embeds_list, True, self.embedding_padding_value)
            padded_labels = pad_sequence(batch_labels_list, True, -100)
            attn_mask = (padded_labels != -100).long()
            
            if padded_embeds.shape[1] == 0:
                 return CausalLMOutputWithPast(loss=torch.tensor(0.0, device=device, requires_grad=True),
                                               logits=torch.empty((0,0,self.audio_vocab_size), device=device))

            gpt2_out = self.gpt2_model(inputs_embeds=padded_embeds, attention_mask=attn_mask, return_dict=True,
                                       output_attentions=output_attentions, output_hidden_states=output_hidden_states)
            logits = self.lm_head(gpt2_out.last_hidden_state)
            loss = self.loss_fn(logits.view(-1, self.audio_vocab_size), padded_labels.view(-1))
            
            return CausalLMOutputWithPast(loss=loss, logits=logits, hidden_states=gpt2_out.hidden_states, attentions=gpt2_out.attentions)
        else: # Inference
            if inputs_embeds is None and input_ids is None: 
                 raise ValueError("inputs_embeds or input_ids required for inference.")
            
            gpt2_args_filtered = {k: v for k, v in {
                "input_ids": input_ids, "inputs_embeds": inputs_embeds, "attention_mask": attention_mask,
                "past_key_values": past_key_values, "use_cache": use_cache, 
                "output_attentions": output_attentions, "output_hidden_states": output_hidden_states,
                "return_dict": True }.items() if v is not None}
            
            gpt2_out = self.gpt2_model(**gpt2_args_filtered) 
            logits = self.lm_head(gpt2_out.last_hidden_state) 
            return CausalLMOutputWithPast(logits=logits, past_key_values=gpt2_out.past_key_values, 
                                          hidden_states=gpt2_out.hidden_states, attentions=gpt2_out.attentions)

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, 
                                      past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, 
                                      **kwargs) -> dict:
        """
        Prepares inputs for Hugging Face .generate().
        Converts input_ids (text or audio) to embeddings.
        """
        attention_mask = kwargs.get("attention_mask")
        prompt_text_length = kwargs.get("prompt_text_length")
        current_embeds: torch.Tensor

        if past_key_values is None: # First step of generation
            text_len = prompt_text_length if prompt_text_length is not None else input_ids.shape[1] - 1
            text_tokens = input_ids[:, :text_len]
            bos_audio_tokens = input_ids[:, text_len:] # Should be BOS_AUDIO_ID

            text_emb = self.gpt2_model.wte(text_tokens)
            bos_emb = self.audio_emb(torch.clamp(bos_audio_tokens, 0, self.audio_vocab_size - 1)) if bos_audio_tokens.numel() > 0 else text_emb # Fallback for safety
            current_embeds = torch.cat([text_emb, bos_emb], dim=1) if bos_audio_tokens.numel() > 0 else text_emb
        else: # Subsequent steps: input_ids is the last generated audio token
            last_audio_tokens = torch.clamp(input_ids[:, -1:], 0, self.audio_vocab_size - 1)
            current_embeds = self.audio_emb(last_audio_tokens)
        
        if attention_mask is not None and past_key_values is not None: # Extend attention mask
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), 
                                       dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
        
        return {"inputs_embeds": current_embeds, "input_ids": None, 
                "past_key_values": past_key_values, "use_cache": kwargs.get("use_cache", self.config.use_cache), 
                "attention_mask": attention_mask}

    def prepare_prompt_for_generation(self, texts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Prepares initial input_ids, attention_mask, and text_token_length for .generate().
        Helper for `generate_speech` in `utils.py`.
        """
        if not texts or not texts[0]: raise ValueError("Input text list is empty or first text is empty.")
        text = texts[0] 
        max_text_tokens = self.config.n_positions // 3 
        text_encoding = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_text_tokens)
        text_token_ids = text_encoding.input_ids.to(device)
        if text_token_ids.numel() == 0: raise ValueError(f"Text '{text[:60]}...' resulted in empty tokens.")
        
        prompt_text_attn_mask = text_encoding.attention_mask.to(device)
        bos_audio_id = torch.tensor([[self.bos_audio_token_id]], dtype=torch.long, device=device)
        
        initial_ids = torch.cat([text_token_ids, bos_audio_id], dim=1)
        initial_attn_mask = torch.cat([prompt_text_attn_mask, torch.ones_like(bos_audio_id)], dim=1)
        
        return initial_ids, initial_attn_mask, text_token_ids.shape[1]

    def decode_audio_tokens(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """
        Decodes flattened Encodec audio tokens into a waveform.
        Assumes audio_tokens are valid codebook indices.
        """
        if audio_tokens.dim() == 1: audio_tokens = audio_tokens.unsqueeze(0)
        batch_size, flat_len = audio_tokens.shape
        if flat_len == 0: return torch.empty(batch_size, 0, device=audio_tokens.device)
        
        # Clamp to codebook_size just in case, though filtering should occur before.
        audio_tokens = torch.clamp(audio_tokens, 0, self.codebook_size - 1)

        # Truncate to be divisible by num_codebooks
        if flat_len % self.num_audio_codebooks != 0:
            flat_len = (flat_len // self.num_audio_codebooks) * self.num_audio_codebooks
            audio_tokens = audio_tokens[:, :flat_len]
            if flat_len == 0: return torch.empty(batch_size, 0, device=audio_tokens.device)
        
        # Reshape: (B, T_flat) -> (B, T_frames, Nq) -> permute to (B, Nq, T_frames)
        audio_codes_reshaped = audio_tokens.reshape(batch_size, -1, self.num_audio_codebooks).permute(0, 2, 1)
        # Encodec expects 4D: (num_chunks=1, B, Nq, T_frames)
        audio_codes_for_decoder = audio_codes_reshaped.unsqueeze(0)
        
        scales = torch.ones((1, batch_size), device=audio_codes_for_decoder.device, dtype=torch.float32)

        with torch.no_grad():
            self.codec.to(audio_codes_for_decoder.device)
            decoded_output = self.codec.decode(audio_codes_for_decoder, audio_scales=scales)
        
        final_audio = decoded_output[0].squeeze(0) # Remove chunk dim
        return final_audio.squeeze(0) if batch_size == 1 and final_audio.dim() > 1 else final_audio
