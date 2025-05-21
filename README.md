# GPT-2 Based Text-to-Speech (TTS) with Mel Spectrogram Prediction

This project implements a Text-to-Speech system using a GPT-2 model to predict mel spectrograms from text. The model is conditioned on reference audio for speaker style transfer, drawing inspiration from XTTS-style conditioning. A pre-trained HiFi-GAN vocoder is used to convert the predicted mel spectrograms into audible waveforms.

## Current Architecture

The system consists of the following key components:

1.  **Text Frontend**: A standard GPT-2 tokenizer processes the input text.
2.  **Reference Audio Conditioning**:
    * A **Conditioning Encoder** (Transformer-based) processes the mel spectrogram of a reference audio clip.
    * A **Perceiver Resampler** distills the output of the Conditioning Encoder into a fixed number of latent embeddings representing the style/speaker characteristics.
3.  **GPT-2 Core Model (`GPT2TTSMelPredictor`)**:
    * The GPT-2 model takes embeddings of the input text and the style latents from the Perceiver.
    * It autoregressively predicts mel spectrogram frames. The final linear layer of the GPT-2 is modified to output `N_MELS` (e.g., 80) values per frame.
    * During training, it uses teacher-forcing with ground truth mel spectrograms.
4.  **Vocoder**: A pre-trained **HiFi-GAN vocoder** (from SpeechBrain) converts the predicted mel spectrogram sequence into an audio waveform.
5.  **Speaker Consistency Loss (SCL) Proxy**: During training, a loss term encourages the style embeddings produced by the Perceiver to be similar to speaker embeddings extracted from the reference audio using a separate pre-trained speaker encoder (e.g., ECAPA-TDNN).

## Current Status & Known Issues

* The model trains and can generate audio.
* The shift from predicting discrete Encodec tokens to continuous mel spectrograms has successfully mitigated the severe mode collapse issues experienced in previous iterations.
* The previous Encodec-based model was also limited to generating very short audio clips (<< 1 second) due to the GPT-2 token limit; the current mel-spectrogram architecture allows for much longer audio generation (e.g., ~9 seconds with current configuration).
* **Audio Quality**: The quality of the generated audio is currently poor. The generated speech defaults to the repetition of the same few sounds or syllables. Generating realistic and intelligible speech is still a work in progress.
* **CER**: The Character Error Rate (CER) calculation in the notebook currently uses a placeholder ASR hypothesis. For a meaningful CER, generated audio needs to be transcribed by an actual ASR system.

## Project Structure

GPT2TTS/
├── .gitignore
├── README.md
├── requirements.txt          # Python package dependencies
├── data/                     # (Needs to be created by user)
│   └── LJSpeech-1.1/         # LJSpeech dataset
│       ├── metadata.csv
│       └── wavs/
├── notebooks/
│   └── gpt2tts.ipynb         # Main Jupyter notebook for training and inference
├── pretrained_models/        # (Created automatically by SpeechBrain/Hugging Face)
│   ├── speechbrain_spkrec-ecapa-voxceleb/ # Cached speaker encoder
│   └── tts-hifigan-ljspeech/ # Cached HiFi-GAN vocoder
├── src/
│   ├── __init__.py           # Makes src a package
│   ├── config.py             # Configuration settings
│   ├── dataset.py            # Dataset loading and preprocessing
│   ├── model.py              # Model definition (GPT2TTSMelPredictor)
│   ├── train.py              # Training script
│   ├── utils.py              # Utility functions (e.g., speech generation)
│   └── evaluation.py         # Evaluation functions (CER, placeholders for UTMOS/SECS)
└── training_runs_mel/        # (Created automatically during training)
    └── training_run_mel_YYYYMMDD-HHMMSS/ # Specific training run
        ├── checkpoints/      # Saved model checkpoints
        ├── logs/             # TensorBoard logs
        └── samples/          # Generated audio samples during validation

## Setup

1.  **Clone the Repository (if applicable)**
2.  **Create a Python Virtual Environment** (recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate 
    ```
3.  **Install Dependencies**:
    It is recommended to install dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    If you need to generate `requirements.txt`, you can use `pip freeze > requirements.txt` after installing packages manually. The key packages include: `torch`, `torchaudio`, `transformers`, `pandas`, `scikit-learn`, `matplotlib`, `soundfile`, `tqdm`, `jiwer`, `speechbrain`, `tensorboard`, and `jupyterlab` (or `jupyter`).

4.  **Download LJSpeech Dataset**:
    * Download the LJSpeech dataset from [its official source](https://keithito.com/LJ-Speech-Dataset/).
    * Extract it and place it in a `data/` directory at the project root, such that you have `GPT2TTS/data/LJSpeech-1.1/`.
    * The path is configured in `src/config.py` (`DATA_DIR`). Update it if you place the dataset elsewhere.

## Running the Code

### 1. Configure Paths
    * Verify that `DATA_DIR` in `src/config.py` points to your `LJSpeech-1.1` dataset directory. The script currently attempts to set this relative to the project root.

### 2. Training
    * The primary way to run training is through the Jupyter notebook: `notebooks/gpt2tts.ipynb`.
    * Open the notebook and run the cells sequentially. Cell 3 ("Run Model Training") executes `src/train.py`.
    * Alternatively, you can run `src/train.py` directly from the terminal (ensure your current working directory is `src/` or adjust paths in `config.py` accordingly, though `config.py` now tries to set paths relative to the project root).
        ```bash
        # From project root:
        python src/train.py 
        ```
    * Training artifacts (checkpoints, logs, samples) will be saved under `training_runs_mel/`.

### 3. Monitoring with TensorBoard
    * During or after training, run TensorBoard to visualize losses and listen to audio samples:
        ```bash
        # From project root:
        tensorboard --logdir training_runs_mel
        ```
    * Open the URL provided (usually `http://localhost:6006/`) in your browser.

### 4. Inference and Evaluation
    * The `notebooks/gpt2tts.ipynb` contains cells (from section 4 onwards) for:
        * Setting up the inference environment.
        * Running a single inference example using the latest/best checkpoint.
        * Performing varied inference (generating multiple samples).
        * Demonstrating calls to evaluation metrics (CER, placeholder UTMOS/SECS).
    * Generated audio samples from the notebook inference cells will be saved into subdirectories within the latest training run folder (e.g., `training_runs_mel/your_run/inference_samples_notebook/`).

## Potential Future Steps

To improve the current system, several avenues can be explored:

* **Mel Prediction Refinement**: Experimenting with different loss functions for mel spectrograms (e.g., adversarial or perceptual losses). Adjusting the GPT-2 architecture or conditioning mechanism for better regression performance.
* **Speaker Consistency**: Enhancing the SCL proxy or implementing a more direct form of speaker consistency training.
* **Autoregressive Generation**: Investigating techniques like scheduled sampling during training or different inference sampling strategies to reduce repetitiveness.
* **Data**: Longer and more diverse training data could significantly benefit generalization.
* **Vocoder**: Exploring different vocoders or fine-tuning the existing HiFi-GAN on the model's predicted mels.
* **Speaker Conditioning**: Considering more sophisticated methods for injecting speaker identity (e.g., adaptive layer normalization).
* **End-of-Sequence Prediction**: Implementing a mechanism for the model to predict the end of mel sequences rather than relying on a fixed maximum length.

