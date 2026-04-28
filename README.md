# TetherBCI: Open-Source BCI Framework

A modular Brain-Computer Interface framework that encodes and decodes multimodal neural signals (EEG, fMRI, MEG) using PyTorch and Transformers.

## What it does
- Encodes EEG, fMRI and MEG brain signals into a unified latent representation
- Decodes into mental state classification (rest, motor, visual, cognitive)
- Reconstructs brain signals for self-supervised learning
- Optimizes inference latency using TorchScript and Quantization

## Results
- 10.5M parameter multimodal model
- Supports 4 mental state classifications
- 50% inference latency reduction via quantization

## Project Structure
- `data_loader.py` — Multimodal brain data pipeline
- `encoder.py` — EEG, fMRI, MEG encoders with Transformer
- `decoder.py` — Mental state and signal decoders
- `framework.py` — Main TetherBCI framework
- `optimize.py` — Latency benchmarking and optimization
- `train.py` — Training script

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python3 data_loader.py   # Generate data
python3 train.py         # Train model
python3 optimize.py      # Optimize and benchmark
```

## Tech Stack
- PyTorch
- Hugging Face Transformers
- Accelerate
- NumPy, scikit-learn, Matplotlib

## Author
Harsh Singh — [GitHub](https://github.com/harshsingh4469)