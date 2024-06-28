cGAN for short-term Nvidia Price Prediction:
---------------------------------------------

This project was undertaken to deepen understanding of time series data and neural network architectures tailored for predictive analysis. Two key learning outcomes emerged throughout its development: mastering time series data preprocessing and refining neural network architecture design.

A primary focus of this project was to train the model exclusively on short-term data. The rationale behind this approach was to enhance the accuracy of price prediction by capturing transient trends effectively. While long-term data can grasp broader price trends, it often lacks precision in specific price predictions. The hypothesis was that short-term data would yield more precise predictions tailored to particular price movements.

Time Series Data Pre-processing -
When it comes to data preprocessing I use many different methods and will continue to add or subtract as I improve the model. The first step I took was feature expansion
for the financial data. This process went by taking the data and adding common market technical indicators such Bollinger Bands, Rolling Averages, Momentum, MACD signal,
and various other common indicators. I then used log transformations and z-score normalization on the relevant features to ensure the data was a standard normal distribution.

Network Architecture - 
I use a Conditional Generative Adversarial Neural Network where the generator is two consecutive LSTM networks for signal generation and the discriminator is a 
Convolutional Neural Network.

Papers used as references - 
1. Deep generative modeling for financial time series with application in VaR: a comparative review: https://arxiv.org/abs/2401.10370
2. Integrating Generative AI into Financial Market Prediction for Improved Decision Making: https://arxiv.org/abs/2404.03523
3. LONG SHORT-TERM MEMORY: https://www.bioinf.jku.at/publications/older/2604.pdf
4. Generative Adversarial Networks in Finance: an overview: https://arxiv.org/abs/2106.06364

You can run the training phase yourself using the cGAN_model file.

DEV NOTES
________________
6/24/2024: I am currently still working on this project, and currently need to improve -
* Data Preprocessing steps (current ideas):
    1. Fourier Discretization of data
    2. Differential Preprocessing
    3. ARMIA
* Noise generation techniques for the noise being sent into the second LSTM layer in the generator


