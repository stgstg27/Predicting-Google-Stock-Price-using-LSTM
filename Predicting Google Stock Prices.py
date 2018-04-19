# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 22:37:44 2018

@author: Saurabh
"""


#Implementing RNN

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout



def read_data(path):
    #Importing training data
    train_data = pd.read_csv(path)
    '''We are only focussing on the open prices of google time series analysis '''
    train_data = train_data.iloc[:,1:2].values
    return train_data

def pre_process_and_split(train_data):
# Feature Scaling
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(train_data)
    
    # Creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    for i in range(60, 1258):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train,sc
    
def build_NN(train_data,Y_train):
    regressor = Sequential()
    #Input Layer and One Lstm Layer
    regressor.add(LSTM(units = 50,return_sequences=True,input_shape = (train_data.shape[1],1)))
    regressor.add(Dropout(0.2))
    #2nd layer
    regressor.add(LSTM(units = 50,return_sequences=True))# , input_shape = (train_data.shape[1],1)))
    regressor.add(Dropout(0.2))
    #3rd layer
    regressor.add(LSTM(units = 50,return_sequences=True)) #, input_shape = (train_data.shape[1],1)))
    regressor.add(Dropout(0.2))
    #4th layer
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    
    #Ouput Layer
    regressor.add(Dense(units = 1))
    
    #Compliling RNN 
    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error' )
    
    return regressor
    
    
        
    
    
def train_NN(regressor,train_data,Y_train,epoch,batchsize):
    regressor.fit(train_data,Y_train,epochs =epoch,batch_size = batchsize) 
    

    
#import training set
train_data = pd.read_csv('Google_Stock_Price_Train.csv')

'''We are only focussing on the open prices of google time series analysis '''
train_data = train_data.iloc[:,1:2].values

    

if __name__ == "__main__":
    print ("Reading Data : Google_Stock_Price_Train")
    train_data1 = read_data('Google_Stock_Price_Train.csv')
    print ("Pre-Processing the Data")
    train_data,Y_train,sc = pre_process_and_split(train_data1)
    print ("Building Neural network Architecture")
    regressor  = build_NN(train_data,Y_train)
    print ("Training Begins")
    train_NN(regressor,train_data,Y_train,100,32)
    
    ###Testing Begins ###
    print ("Reading Data : Google_Stock_Price_Test")
    dataaet_test = read_data('Google_Stock_Price_Test.csv')
    print ("Pre-Processing for the test Data")
        # Getting the real stock price of 2017
    dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
    real_stock_price = dataset_test.iloc[:, 1:2].values
    
    # Getting the predicted stock price of 2017
    orig_train_data = pd.read_csv('Google_Stock_Price_Train.csv')
    dataset_total = pd.concat((orig_train_data['Open'], dataset_test['Open']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 80):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    # Visualising the results
    plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.show()
    
    
    
    
    
    
    

