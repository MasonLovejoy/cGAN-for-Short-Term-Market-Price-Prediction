cGAN for short-term Nvidia Price Prediction:

This project was used to learn more about time series data and the neural network architectures used to predict this data. When going through this project there were
two main learning opportunities, the first being time series data preprocessing and the second being network architecture. I also specifically wanted to train this model
using only short-term data in hopes that it could capture the temporary trend for accurate price prediction. My thought is that long-term data prediction is good for
capturing the trend of the price movement but bad for specific price prediction. I hope that providing only short-term data would be better for price-specific 
prediction.

Time Series Data Pre-processing -
When it comes to data preprocessing I use many different methods and will continue to add or subtract as I improve the model. The first step I took was feature expansion
for the financial data. This process went by taking the data and adding common market technical indicators such Bollinger Bands, Rolling Averages, Momentum, MACD signal,
and various other common indicators. I then used log transformations and z-score normalization on the relevant features to ensure the data was a standard normal distribution.

Network Architecture - 
I use a Conditional Generative Adversarial Neural Network where the generator is two consecutive LSTM networks for signal generation and the discriminator is a 
Convolutional Neural Network.
