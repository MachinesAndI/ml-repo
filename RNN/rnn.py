import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

training_set = pd.read_csv("Google_Stock_Price_Train.csv")
# Taking the opening values of the stock
training_set = training_set.iloc[:,1:2].values

#Feature Scaling(Applying normalization)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
training_set = scaler.fit_transform(training_set)

# Getting the inputs and the outputs
X_train = training_set[0:1257]
y_train = training_set[1:1258]

# Reshaping the data 3D tensor with shape (batch_size, timesteps, input_dim)
X_train = np.reshape(X_train,(1257,1,1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

regressor = Sequential()


regressor.add(LSTM(units = 4,activation='sigmoid', input_shape = (None,1))) #input shape args first- no. 
#timestamps we have 1 so either put 1 or None to add autmatically and second is no. of input features

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train,y_train,batch_size =32, epochs =200) 

test_set = pd.read_csv("Google_Stock_Price_Test.csv")
# Taking the opening values of the stock
real_stock_prices = test_set.iloc[:,1:2].values

inputs= real_stock_prices
inputs = scaler.transform(inputs)
inputs = np.reshape(inputs,(20,1,1))
predicted_values = regressor.predict(inputs)
predicted_values = scaler.inverse_transform(predicted_values)

# Visualising results
plt.plot(real_stock_prices, color='red', label='Real stock prices')
plt.plot(predicted_values, color='blue', label='Predicted stock prices')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock prices')
plt.legend()
plt.show()

# Predicting the training dataset stock prices 2012-2016
real_stock_prices_train = pd.read_csv("Google_Stock_Price_Train.csv")
real_stock_prices_train = real_stock_prices_train.iloc[:,1:2].values

predicted_stock_prices_train = regressor.predict(X_train)
predicted_stock_prices_train = scaler.inverse_transform(predicted_stock_prices_train)

plt.plot(real_stock_prices_train, color='red', label='Real stock prices')
plt.plot(predicted_stock_prices_train, color='blue', label='Predicted stock prices')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock prices')
plt.legend()
plt.show()

# Evaluating the performance of RNN
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_prices,predicted_values))