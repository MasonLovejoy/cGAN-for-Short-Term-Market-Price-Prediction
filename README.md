# Conditional GAN for Multivariate Stock Time-Series Forecasting

This project implements a Conditional GAN (cGAN) architecture for multistep forecasting of stock price sequences, built from scratch in PyTorch. The system learns to generate realistic continuations of market windows using adversarial, reconstruction, and feature-matching losses.

## The project includes:

- full training framework + logging + checkpointing
- custom preprocessing pipeline for technical indicators
- custom discriminator + encoder/decoder generator
- Exponential Moving Average of weights for stability
- gradient penalty + label smoothing + instance noise

## Motivation

Traditional time-series models struggle to capture multimodal futures and uncertainty. GAN-based forecasters can learn distributions over future price movements rather than point predictions, enabling more realistic simulations of market trajectories.

# Project Structure
├── data_preprocessing.py   # feature engineering + tensor construction  
├── models.py               # generator/discriminator modules + EMA  
├── train.py                # training system + checkpointing + metrics  
├── checkpoints/            # saved models  
├── logs/                   # metrics, visualizations, and summaries  
└── data/                   # raw + processed csv and tensor data  


# Usage
Train the cGAN from CLI:  
python train.py  

Checkpoints and logs will be automatically stored in:  

./checkpoints/  
./logs/  

# Output artifacts
- learning curves (.png)  
- JSON training summary files  
- saved checkpoints & best model
- console + file logging

# Papers used as references - 
1. Deep generative modeling for financial time series with application in VaR: a comparative review: https://arxiv.org/abs/2401.10370
2. Integrating Generative AI into Financial Market Prediction for Improved Decision Making: https://arxiv.org/abs/2404.03523
3. LONG SHORT-TERM MEMORY: https://www.bioinf.jku.at/publications/older/2604.pdf
4. Generative Adversarial Networks in Finance: an overview: https://arxiv.org/abs/2106.06364


