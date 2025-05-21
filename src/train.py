# src/train.py
"""
Main training script for the GPT2TTSMelPredictor model.
Handles model initialization, data loading, the training loop,
validation, checkpointing, and TensorBoard logging.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import os
import time
import numpy as np
from tqdm import tqdm
import random
import soundfile as sf

import config as train_config
from model import GPT2TTSMelPredictor
from dataset import get_data_loaders_mel 
from utils import generate_speech_mel 

def set_seed(seed_value: int):
    """ Sets the random seed for reproducibility across libraries. """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # These can slow down training, use for debugging or final runs if needed
        # torch.backends.cudnn.deterministic = True 
        # torch.backends.cudnn.benchmark = False

def main_train():
    """ Main function to orchestrate the training process. """
    set_seed(train_config.SEED)
    device = train_config.DEVICE
    
    # --- Setup Output Directories ---
    run_name = f"training_run_mel_{time.strftime('%Y%m%d-%H%M%S')}"
    # OUTPUT_DIR is now project_root/training_runs_mel from config
    current_output_dir = os.path.join(train_config.OUTPUT_DIR, run_name) 
    checkpoints_dir = os.path.join(current_output_dir, "checkpoints")
    logs_dir = os.path.join(current_output_dir, "logs") # For TensorBoard
    samples_dir = os.path.join(current_output_dir, "samples") # For generated audio samples
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=logs_dir)
    print(f"Training run output will be saved in: {current_output_dir}")
    print(f"Using device: {device}")

    # --- Data Loaders ---
    # Max lengths are now taken from config and passed to get_data_loaders_mel
    train_loader, val_loader, gpt2_tokenizer = get_data_loaders_mel(
        gpt2_tokenizer_name_or_path=train_config.GPT2_MODEL_NAME,
        batch_size=train_config.BATCH_SIZE,
        num_workers=train_config.NUM_WORKERS,
        pin_memory=train_config.PIN_MEMORY,
        train_split_ratio=train_config.TRAIN_SPLIT_RATIO,
        max_text_len=train_config.MAX_TEXT_LEN_DATASET, 
        max_mel_frames=train_config.MAX_MEL_FRAMES_DATASET, 
        use_curriculum=train_config.USE_CURRICULUM_LEARNING,
        # Initial curriculum length, or None if not using curriculum
        current_max_mel_frames_curriculum=train_config.CURRICULUM_START_MAX_MEL_FRAMES if train_config.USE_CURRICULUM_LEARNING else None
    )
    
    # --- Model Initialization ---
    model = GPT2TTSMelPredictor.from_pretrained_custom(
        tokenizer=gpt2_tokenizer, 
        device=device
        # checkpoint_path can be passed here to resume, but typically handled by loading best/latest later
    )
    
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel): # Avoid re-wrapping
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)
    model.to(device)

    # --- Optimizer and Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=train_config.LEARNING_RATE, 
                            weight_decay=train_config.WEIGHT_DECAY, eps=train_config.ADAM_EPSILON)
    
    # Total training steps for scheduler
    num_training_steps = len(train_loader) * train_config.NUM_EPOCHS // train_config.GRADIENT_ACCUMULATION_STEPS
    
    if train_config.LR_SCHEDULER_TYPE == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=train_config.WARMUP_STEPS, 
                                                    num_training_steps=num_training_steps)
    elif train_config.LR_SCHEDULER_TYPE == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=train_config.WARMUP_STEPS,
                                                    num_training_steps=num_training_steps)
    else: # "constant" or None
        scheduler = None

    # --- Training Loop ---
    global_step = 0
    best_val_loss = float('inf')
    
    # Initialize curriculum current max frames for the training loop
    current_max_frames_for_curriculum_in_loop = train_config.CURRICULUM_START_MAX_MEL_FRAMES \
        if train_config.USE_CURRICULUM_LEARNING else train_config.MAX_MEL_FRAMES_DATASET

    for epoch in range(train_config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{train_config.NUM_EPOCHS} ---")
        
        # Update dataset's curriculum parameter if applicable (affects next epoch's data loading if not dynamic)
        if train_config.USE_CURRICULUM_LEARNING and hasattr(train_loader.dataset, 'current_max_mel_frames_for_curriculum'):
            train_loader.dataset.current_max_mel_frames_for_curriculum = current_max_frames_for_curriculum_in_loop
            print(f"Curriculum: Max mel frames for dataset set to {current_max_frames_for_curriculum_in_loop}")

        model.train()
        epoch_mel_loss_sum = 0.0
        epoch_scl_loss_sum = 0.0
        epoch_total_loss_sum = 0.0
        num_batches_in_epoch = 0
        
        optimizer.zero_grad() # Zero gradients at the start of each epoch / accumulation cycle

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Training")
        for batch_idx, batch in progress_bar:
            if not batch: # Handle empty batch from collate_fn if all items were skipped
                print(f"Skipping empty batch at index {batch_idx} in epoch {epoch+1}")
                continue
            num_batches_in_epoch +=1

            text_input_ids = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            target_mel = batch["target_mel_spectrogram"].to(device)
            teacher_forcing_mel = batch["teacher_forcing_mel_spectrogram"].to(device)
            mel_actual_frame_mask = batch["mel_actual_frame_mask"].to(device) 
            ref_audio_wav = batch["reference_audio_waveform"].to(device)
            ref_audio_sr = batch["reference_audio_sample_rate"] # Scalar, from first item

            # Forward pass
            model_outputs = model(
                text_input_ids=text_input_ids,
                reference_audio_waveform=ref_audio_wav, 
                reference_audio_sample_rate=ref_audio_sr,
                target_mel_spectrogram=target_mel,
                input_mel_spectrogram_shifted=teacher_forcing_mel, 
                text_attention_mask=text_attention_mask,
                mel_actual_frame_mask=mel_actual_frame_mask 
            )
            
            mel_loss = model_outputs["mel_loss"]
            scl_proxy_loss = model_outputs["scl_proxy_loss"]
            
            current_total_loss = mel_loss + scl_proxy_loss * train_config.SCL_PROXY_WEIGHT
            
            # Normalize loss for gradient accumulation
            accumulated_loss = current_total_loss / train_config.GRADIENT_ACCUMULATION_STEPS
            accumulated_loss.backward()

            # Accumulate per-batch (non-normalized by grad_accum) losses for epoch average
            epoch_mel_loss_sum += mel_loss.item()
            epoch_scl_loss_sum += scl_proxy_loss.item()
            epoch_total_loss_sum += current_total_loss.item()


            if (batch_idx + 1) % train_config.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.MAX_GRAD_NORM)
                optimizer.step()
                if scheduler: 
                    scheduler.step()
                optimizer.zero_grad() # Zero gradients after optimizer step
                global_step += 1

                # Log training losses (current batch's effective loss)
                if global_step % train_config.LOG_INTERVAL == 0:
                    writer.add_scalar("Loss_Batch/train_mel", mel_loss.item(), global_step)
                    writer.add_scalar("Loss_Batch/train_scl_proxy", scl_proxy_loss.item(), global_step)
                    writer.add_scalar("Loss_Batch/train_total", current_total_loss.item(), global_step)
                    writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], global_step)
                    progress_bar.set_postfix({
                        "mel_loss": f"{mel_loss.item():.4f}", 
                        "scl_loss": f"{scl_proxy_loss.item():.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                    })

                # --- Validation and Checkpointing (based on global_step) ---
                if global_step > 0 and global_step % train_config.VALIDATION_INTERVAL == 0:
                    model.eval()
                    val_mel_loss_sum = 0.0
                    val_scl_loss_sum = 0.0
                    val_total_loss_sum = 0.0
                    num_val_batches = 0
                    with torch.no_grad():
                        val_progress_bar = tqdm(val_loader, total=len(val_loader), desc=f"Step {global_step} Validation", leave=False)
                        for val_batch_idx, val_batch in enumerate(val_progress_bar):
                            if not val_batch: continue # Skip empty val batches
                            num_val_batches += 1
                            v_txt_ids = val_batch["text_input_ids"].to(device)
                            v_txt_mask = val_batch["text_attention_mask"].to(device)
                            v_tgt_mel = val_batch["target_mel_spectrogram"].to(device)
                            v_teacher_mel = val_batch["teacher_forcing_mel_spectrogram"].to(device)
                            v_mel_mask = val_batch["mel_actual_frame_mask"].to(device)
                            v_ref_wav = val_batch["reference_audio_waveform"].to(device)
                            v_ref_sr = val_batch["reference_audio_sample_rate"]

                            val_model_outputs = model(
                                text_input_ids=v_txt_ids, reference_audio_waveform=v_ref_wav,
                                reference_audio_sample_rate=v_ref_sr, target_mel_spectrogram=v_tgt_mel,
                                input_mel_spectrogram_shifted=v_teacher_mel, text_attention_mask=v_txt_mask,
                                mel_actual_frame_mask=v_mel_mask
                            )
                            val_mel_loss = val_model_outputs["mel_loss"]
                            val_scl_proxy_loss = val_model_outputs["scl_proxy_loss"]
                            val_current_total_loss = val_mel_loss + val_scl_proxy_loss * train_config.SCL_PROXY_WEIGHT

                            val_mel_loss_sum += val_mel_loss.item()
                            val_scl_loss_sum += val_scl_proxy_loss.item()
                            val_total_loss_sum += val_current_total_loss.item()
                            val_progress_bar.set_postfix({"val_mel_loss": f"{val_mel_loss.item():.4f}"})
                    
                    avg_val_mel_loss = val_mel_loss_sum / num_val_batches if num_val_batches > 0 else 0
                    avg_val_scl_loss = val_scl_loss_sum / num_val_batches if num_val_batches > 0 else 0
                    avg_val_total_loss = val_total_loss_sum / num_val_batches if num_val_batches > 0 else 0
                    
                    writer.add_scalar("Loss_Val/mel", avg_val_mel_loss, global_step)
                    writer.add_scalar("Loss_Val/scl_proxy", avg_val_scl_loss, global_step)
                    writer.add_scalar("Loss_Val/total", avg_val_total_loss, global_step)
                    print(f"\nValidation @ Step {global_step}: Total Loss: {avg_val_total_loss:.4f}, Mel Loss: {avg_val_mel_loss:.4f}, SCL Proxy: {avg_val_scl_loss:.4f}")

                    # Save checkpoints
                    save_path_latest = os.path.join(checkpoints_dir, "latest_model_checkpoint.pth")
                    torch.save(model.state_dict(), save_path_latest)
                    if avg_val_total_loss < best_val_loss:
                        best_val_loss = avg_val_total_loss
                        save_path_best = os.path.join(checkpoints_dir, "best_val_loss_model.pth")
                        torch.save(model.state_dict(), save_path_best)
                        print(f"Best val loss improved to {best_val_loss:.4f}. Checkpoint saved: {save_path_best}")
                    
                    # Log a generated audio sample
                    try:
                        sample_text_val = "This is a validation audio sample."
                        # Use a reference audio from the current validation batch
                        if val_loader.dataset and len(val_loader.dataset) > 0 and val_batch: 
                             val_ref_audio_for_sample = val_batch["reference_audio_waveform"][0].cpu() 
                             val_ref_sr_for_sample = val_batch["reference_audio_sample_rate"] 

                             generated_waveform_np_val = generate_speech_mel(
                                 model.module if hasattr(model, 'module') else model,
                                 sample_text_val, val_ref_audio_for_sample, val_ref_sr_for_sample, device,
                                 max_mel_frames=train_config.LOGGING_MAX_MEL_FRAMES
                             )
                             if generated_waveform_np_val is not None:
                                 sample_filename_val = os.path.join(samples_dir, f"sample_step{global_step}.wav")
                                 sf.write(sample_filename_val, generated_waveform_np_val, train_config.TARGET_SAMPLE_RATE)
                                 writer.add_audio(f"AudioSample_Val/step{global_step}", generated_waveform_np_val, 
                                                  global_step, sample_rate=train_config.TARGET_SAMPLE_RATE)
                                 print(f"Logged validation audio sample: {sample_filename_val}")
                        else:
                            print("Skipping validation audio sample generation (val_batch empty or problematic).")
                    except Exception as e_gen_val:
                        print(f"Error during validation sample generation: {e_gen_val}")
                    model.train() # Switch back to training mode

                # Update curriculum learning max frames
                if train_config.USE_CURRICULUM_LEARNING and \
                   current_max_frames_for_curriculum_in_loop < train_config.CURRICULUM_END_MAX_MEL_FRAMES and \
                   global_step > 0 and global_step % train_config.CURRICULUM_INCREMENT_STEPS == 0:
                    current_max_frames_for_curriculum_in_loop = min(
                        current_max_frames_for_curriculum_in_loop + train_config.CURRICULUM_INCREMENT_AMOUNT,
                        train_config.CURRICULUM_END_MAX_MEL_FRAMES
                    )
                    # Update the dataset instance directly for the next epoch's item fetching
                    if hasattr(train_loader.dataset, 'current_max_mel_frames_for_curriculum'):
                        train_loader.dataset.current_max_mel_frames_for_curriculum = current_max_frames_for_curriculum_in_loop
                        print(f"Curriculum Update @ Step {global_step}: Max mel frames for dataset set to {current_max_frames_for_curriculum_in_loop}")
        
        # End of epoch logging
        avg_epoch_mel_loss = epoch_mel_loss_sum / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
        avg_epoch_scl_loss = epoch_scl_loss_sum / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
        avg_epoch_total_loss = epoch_total_loss_sum / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
        writer.add_scalar("Loss_Epoch/train_mel", avg_epoch_mel_loss, epoch + 1)
        writer.add_scalar("Loss_Epoch/train_scl_proxy", avg_epoch_scl_loss, epoch + 1)
        writer.add_scalar("Loss_Epoch/train_total", avg_epoch_total_loss, epoch + 1)
        print(f"--- Epoch {epoch+1} Complete. Avg Total Loss: {avg_epoch_total_loss:.4f}, Avg Mel Loss: {avg_epoch_mel_loss:.4f}, Avg SCL Proxy: {avg_epoch_scl_loss:.4f} ---")

    writer.close()
    print("--- Training Finished ---")

if __name__ == '__main__':
    # This structure allows running train.py directly.
    # Ensure DATA_DIR in config.py is correctly set.
    if not os.path.exists(train_config.DATA_DIR) or \
       not os.path.exists(train_config.METADATA_FILE) or \
       not os.path.exists(train_config.WAVS_DIR):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"! ERROR: Dataset path not correctly configured in src/config.py.             !")
        print(f"! Please ensure DATA_DIR ('{train_config.DATA_DIR}') and its contents exist.   !")
        print(f"! Specifically, METADATA_FILE: '{train_config.METADATA_FILE}'")
        print(f"! and WAVS_DIR: '{train_config.WAVS_DIR}' must be valid paths.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        main_train()
