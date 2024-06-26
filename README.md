cGAN for short-term Nvidia Price Prediction:

This project was undertaken to deepen understanding of time series data and neural network architectures tailored for predictive analysis. Throughout its development, two key learning outcomes emerged: mastering time series data preprocessing and refining neural network architecture design.

A primary focus of this project was to train the model exclusively on short-term data. The rationale behind this approach was to enhance the accuracy of price prediction by capturing transient trends effectively. While long-term data can grasp broader price trends, it often lacks precision in specific price predictions. The hypothesis here was that utilizing short-term data would yield more precise predictions tailored to specific price movements.


Time Series Data Pre-processing -
When it comes to data preprocessing I use many different methods and will continue to add or subtract as I improve the model. The first step I took was feature expansion
for the financial data. This process went by taking the data and adding common market technical indicators such Bollinger Bands, Rolling Averages, Momentum, MACD signal,
and various other common indicators. I then used log transformations and z-score normalization on the relevant features to ensure the data was a standard normal distribution.

Network Architecture - 
I use a Conditional Generative Adversarial Neural Network where the generator is two consecutive LSTM networks for signal generation and the discriminator is a 
Convolutional Neural Network.
