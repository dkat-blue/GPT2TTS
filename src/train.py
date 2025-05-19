# src/train.py
"""
Main script for training the GPT2TTS model.
Handles dataset loading, model initialization, training/validation loops, and logging.
"""
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, AutoProcessor, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import time
import datetime
import os
import numpy as np
from tqdm import tqdm 

import config 
from model import GPT2TTS
from dataset import get_dataloaders 
from utils import log_audio_samples

def main_train():
    """Orchestrates the GPT2TTS model training and validation pipeline."""
    config.create_run_directories()
    print(f"Using device: {config.DEVICE}")

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(config.GPT2_MODEL_NAME)
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_tokenizer.pad_token_id = gpt2_tokenizer.eos_token_id

    encodec_processor = AutoProcessor.from_pretrained(config.ENCODEC_MODEL_NAME)

    print("Preparing DataLoaders...")
    train_dataloader, val_dataloader = get_dataloaders(
        encodec_processor, gpt2_tokenizer, 
        config.VALIDATION_SPLIT_RATIO, config.BATCH_SIZE
    )
    if not train_dataloader:
        print("No training data. Exiting.")
        return

    print("Initializing GPT2TTS model...")
    tts_model = GPT2TTS(tokenizer=gpt2_tokenizer).to(config.DEVICE)
    
    optimizer = AdamW(tts_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    num_training_steps = len(train_dataloader) * config.NUM_EPOCHS
    print(f"Total training steps: {num_training_steps}")

    lr_scheduler = None
    if config.LR_SCHEDULER_TYPE == "linear_with_warmup":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, config.NUM_WARMUP_STEPS, num_training_steps
        )
        print(f"Using linear LR scheduler: {config.NUM_WARMUP_STEPS} warmup steps, {num_training_steps} total.")

    all_train_batch_losses, epoch_train_losses, epoch_val_losses = [], [], []
    best_val_loss, epochs_no_improve = float('inf'), 0
    
    current_curriculum_audio_len = config.CURRICULUM_INITIAL_AUDIO_TOKEN_LEN if config.USE_CURRICULUM_LEARNING else None
    if config.USE_CURRICULUM_LEARNING:
        print(f"Curriculum learning enabled: initial audio token length = {current_curriculum_audio_len}")
    
    start_time_total = time.time()
    print(f"\n--- Starting Training for {config.NUM_EPOCHS} Epochs ---")
    
    epochs_pbar = tqdm(range(config.NUM_EPOCHS), desc="Epochs", unit="epoch", dynamic_ncols=True)
    
    for epoch in epochs_pbar: 
        epoch_start_time = time.time()
        epochs_pbar.set_description(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # --- Training Phase ---
        tts_model.train()
        current_epoch_train_batch_losses = []
        num_successful_batches = 0

        if config.USE_CURRICULUM_LEARNING:
            if epoch > 0 and (epoch % config.CURRICULUM_INCREMENT_INTERVAL_EPOCHS == 0):
                 current_curriculum_audio_len += config.CURRICULUM_INCREMENT_AUDIO_TOKEN_LEN
                 practical_max_len = tts_model.config.n_positions - 50 
                 current_curriculum_audio_len = min(current_curriculum_audio_len, practical_max_len) 
                 tqdm.write(f"Curriculum Update: Max audio token length now {current_curriculum_audio_len}")
            elif epoch == 0 and config.CURRICULUM_INCREMENT_INTERVAL_EPOCHS == 1:
                tqdm.write(f"Curriculum: Max audio token length for epoch 1 is {current_curriculum_audio_len}")

        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = tts_model(
                texts=batch['texts'], audio_waveforms=batch['audio_waveforms'], 
                return_loss=True, return_dict=True,
                current_max_audio_tokens_for_curriculum=current_curriculum_audio_len
            )
            loss = outputs.loss 
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                if loss is not None: tqdm.write(f"Warning: Invalid training loss epoch {epoch+1}, batch {i+1}.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(tts_model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            if lr_scheduler: lr_scheduler.step()
            
            current_epoch_train_batch_losses.append(loss.item())
            num_successful_batches +=1
            if num_successful_batches > 0:
                epochs_pbar.set_postfix({
                    'Train Loss': f"{np.mean(current_epoch_train_batch_losses):.4f}",
                    'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
        
        avg_train_loss = np.mean(current_epoch_train_batch_losses) if current_epoch_train_batch_losses else float('nan') 
        epoch_train_losses.append(avg_train_loss)
        tqdm.write(f"Epoch {epoch+1} Train Summary: Avg Loss={avg_train_loss:.4f} ({num_successful_batches}/{len(train_dataloader)} batches)")

        # --- Validation Phase ---
        avg_val_loss = float('nan')
        if val_dataloader and num_successful_batches > 0:
            tts_model.eval()
            current_epoch_val_losses_list = [] # Store batch losses for current val epoch
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_outputs = tts_model(
                        texts=val_batch['texts'], audio_waveforms=val_batch['audio_waveforms'],
                        return_loss=True, return_dict=True,
                        current_max_audio_tokens_for_curriculum=current_curriculum_audio_len
                    )
                    val_loss_item = val_outputs.loss
                    if val_loss_item is not None and not (torch.isnan(val_loss_item) or torch.isinf(val_loss_item)):
                        current_epoch_val_losses_list.append(val_loss_item.item())
                        if current_epoch_val_losses_list: # Check if list is not empty
                             epochs_pbar.set_postfix({
                                'Train Loss': f"{avg_train_loss:.4f}", 
                                'Val Loss': f"{np.mean(current_epoch_val_losses_list):.4f}", # Running avg val loss
                                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
                            })
            avg_val_loss = np.mean(current_epoch_val_losses_list) if current_epoch_val_losses_list else float('nan')
            epoch_val_losses.append(avg_val_loss)
            tqdm.write(f"Epoch {epoch+1} Val Summary: Avg Loss={avg_val_loss:.4f}")
        else:
            epoch_val_losses.append(float('nan')) # Record NaN if no validation
            if not val_dataloader: tqdm.write(f"Epoch {epoch+1}: No validation dataloader.")
            else: tqdm.write(f"Epoch {epoch+1}: Skipping validation (no successful train batches).")

        # --- Checkpointing & Early Stopping ---
        if num_successful_batches > 0:
            torch.save(tts_model.state_dict(), config.LATEST_MODEL_CHECKPOINT_FILE)
            loss_for_comp = avg_val_loss if val_dataloader and not np.isnan(avg_val_loss) else avg_train_loss
            if not np.isnan(loss_for_comp) and loss_for_comp < best_val_loss - config.MIN_DELTA_IMPROVEMENT:
                best_val_loss = loss_for_comp
                torch.save(tts_model.state_dict(), config.BEST_MODEL_VAL_CHECKPOINT_FILE)
                metric_name = "val_loss" if val_dataloader and not np.isnan(avg_val_loss) else "train_loss"
                tqdm.write(f"Saved best model ({metric_name}: {best_val_loss:.4f})")
                epochs_no_improve = 0
            elif not np.isnan(loss_for_comp): epochs_no_improve += 1
        
        epoch_duration = str(datetime.timedelta(seconds=int(time.time() - epoch_start_time)))
        final_postfix = {'Train Loss': f"{avg_train_loss:.4f}" if not np.isnan(avg_train_loss) else "N/A",
                         'LR': f"{optimizer.param_groups[0]['lr']:.2e}", 'Epoch Time': epoch_duration}
        if not np.isnan(avg_val_loss): final_postfix['Val Loss'] = f"{avg_val_loss:.4f}"
        else: final_postfix['Val Loss'] = "N/A"
        epochs_pbar.set_postfix(final_postfix)

        log_audio_samples(tts_model, epoch + 1, config.DEVICE)
        tqdm.write(f"Epoch {epoch+1} duration: {epoch_duration}\n") 

        if val_dataloader and epochs_no_improve >= config.EARLY_STOPPING_PATIENCE and not np.isnan(avg_val_loss):
            tqdm.write(f"Early stopping after {epoch+1} epochs.")
            break
    epochs_pbar.close()
            
    print(f"--- Training Finished --- Total time: {datetime.timedelta(seconds=int(time.time() - start_time_total))}")

    # --- Plotting ---
    if all_train_batch_losses:
        plt.figure(figsize=(12,6)); plt.plot(all_train_batch_losses, label="Train Batch Loss")
        plt.title('Train Batch Loss Over Steps'); plt.xlabel('Batch Num'); plt.ylabel('Loss'); plt.grid(True); plt.legend()
        try: plt.savefig(config.TRAIN_LOSS_CHART_FILE); print(f"Saved: {config.TRAIN_LOSS_CHART_FILE}")
        except Exception as e: print(f"Error saving train loss chart: {e}"); plt.close()

    train_losses_plot = [l for l in epoch_train_losses if not np.isnan(l)]
    val_losses_plot = [l for l in epoch_val_losses if not np.isnan(l)]
    if train_losses_plot:
        plt.figure(figsize=(12,6)); plt.plot(range(len(train_losses_plot)), train_losses_plot, label="Avg Train Loss/Epoch", marker='o')
        if val_losses_plot: plt.plot(range(len(val_losses_plot)), val_losses_plot, label="Avg Val Loss/Epoch", marker='x')
        plt.title('Avg Train & Val Loss Over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Avg Loss')
        plt.xticks(range(max(len(train_losses_plot), len(val_losses_plot) if val_losses_plot else 0))) 
        plt.grid(True); plt.legend()
        try: plt.savefig(config.VAL_LOSS_CHART_FILE); print(f"Saved: {config.VAL_LOSS_CHART_FILE}")
        except Exception as e: print(f"Error saving val loss chart: {e}"); plt.close()

    # --- Final Logging ---
    with open(config.TRAINING_LOG_FILE, "w") as f:
        f.write(f"Training Run: {config.RUN_ID} @ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Epochs Planned: {config.NUM_EPOCHS}, Run: {epoch + 1}\n")
        f.write(f"Samples: {config.MAX_TOTAL_SAMPLES or 'All'}, Val Ratio: {config.VALIDATION_SPLIT_RATIO}\n")
        f.write(f"LR: {config.LEARNING_RATE}, Weight Decay: {config.WEIGHT_DECAY}, Label Smooth: {config.LABEL_SMOOTHING}\n")
        f.write(f"Batch Size: {config.BATCH_SIZE}, Device: {config.DEVICE}\n")
        f.write(f"Curriculum: {config.USE_CURRICULUM_LEARNING}, Initial Len: {config.CURRICULUM_INITIAL_AUDIO_TOKEN_LEN if config.USE_CURRICULUM_LEARNING else 'N/A'}\n")
        best_loss_str = f"{best_val_loss:.4f}" if best_val_loss != float('inf') else "N/A"
        f.write(f"Best Val/Train Loss: {best_loss_str}\n")
        final_train_str = f"{epoch_train_losses[-1]:.4f}" if epoch_train_losses and not np.isnan(epoch_train_losses[-1]) else "N/A"
        f.write(f"Final Avg Train Loss: {final_train_str}\n")
    print(f"Training summary saved to {config.TRAINING_LOG_FILE}")

if __name__ == '__main__':
    main_train()
