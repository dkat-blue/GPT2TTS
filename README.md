# LLM-based Text-to-Speech (TTS) Project

A project to implement and experiment with a Text-to-Speech system using a Large Language Model (GPT-2) architecture combined with an Encodec audio codec.

## Overview

This project aims to:
- Take text input.
- Use a GPT-2 based model to predict Encodec audio tokens.
- Decode these tokens back into an audio waveform using the Encodec decoder.

The codebase is structured into:
- `src/`: Contains the core Python modules for configuration, dataset handling, model definition, training, and utilities.
- `notebooks/`: Jupyter notebooks for experimentation, training initiation, and inference.
- `data/`: (Intended for storing datasets like LJSpeech - currently ignored by git).
- `training_runs/`: (Default output directory for model checkpoints, logs, and generated audio samples during training - ignored by git).

## Setup

1.  **Clone the repository.**
2.  **Create a Python virtual environment** and activate it:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Key libraries include: `torch`, `torchaudio`, `transformers`, `pandas`, `scikit-learn`, `matplotlib`, `soundfile`, `tqdm`, `librosa`.
4.  **Download Dataset:** Place the LJSpeech-1.1 dataset (or other compatible dataset) into the `data/` directory. Update paths in `src/config.py` if necessary.
5.  **Run Training:** Execute the `llm_tts_main.ipynb` notebook or run the `src/train.py` script directly.

## Usage

-   **Training:** The main training process is orchestrated by `src/train.py`, which can be called from the `llm_tts_main.ipynb` notebook. Configuration is managed in `src/config.py`.
-   **Inference:** The notebook also provides an example of how to load a trained checkpoint and perform inference.

---
*This README is a basic placeholder and can be expanded with more details on the architecture, training results, and usage examples.*
