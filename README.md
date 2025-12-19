# Noise-Conditioned GAN for Quantum Circuit V&V

This project explores machine learning–assisted quantum circuit verification and validation (QCVV) by learning to model and forecast measurement outcome distributions under realistic noise.
We build a Qiskit-based dataset generator that sweeps over circuit depths and noise strengths, then train a conditional GAN to predict how distribution drift evolves across future timesteps.

## Project Overview
### 1. Dataset Generation (Qiskit Aer)  
We simulate noisy 3-qubit random circuits across a grid of:  
- depth ∈ 1–80  
- noise levels ∈ 0.0–0.2  
- 2048 measurement shots per circuit  
- amplitude damping + depolarizing models  

Output shape:
[num_experiments, sequence_length=4, feature_dim = conditioning + distribution]
Tensor is saved to disk for training.

### 2. Conditional GAN Architecture (PyTorch)
The model consists of:
- Generator  
- 3-layer LSTM encoder/decoder  
- Autoregressive conditioning on past sequence  
- Noise-augmented decoding  
- MLP output head w/ Tanh activation  
- Discriminator  
- Spectral normalization  

### 3. Training Loop + Logging
- Adversarial D/G optimization  
- Checkpoints + best model saving  
- Per-iteration averaged metrics  
- Smoothed curve plotting  
- Run summaries saved to JSON

## Repository Structure  
├── data_preprocessing.py     # Qiskit-based distribution dataset generation  
├── models.py                 # Generator + Discriminator modules + EMA helper  
├── train.py                  # full GAN training loop & logging utilities  
├── data/                     # generated tensors (not included)  
├── checkpoints/              # saved model states  
├── logs/                     # metric logs + summaries  
└── README.md

## Usage
Generate dataset  
python data_preprocessing.py  

Train model  
python train.py  

## Future Work
- expand qubit count + entanglement topology sweeps  
- learn noise transfer functions for device drift modeling  
- integrate with cloud hardware QCVV experiments   

# Papers used as references - 
1. Deep generative modeling for financial time series with application in VaR: a comparative review: https://arxiv.org/abs/2401.10370
2. Integrating Generative AI into Financial Market Prediction for Improved Decision Making: https://arxiv.org/abs/2404.03523
3. LONG SHORT-TERM MEMORY: https://www.bioinf.jku.at/publications/older/2604.pdf
4. Generative Adversarial Networks in Finance: an overview: https://arxiv.org/abs/2106.06364


