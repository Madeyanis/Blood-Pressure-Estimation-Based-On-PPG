import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, BatchNormalization, AveragePooling1D, Activation, Conv1D, Dropout, MaxPooling1D, Flatten, Dense, LSTM, Reshape, GlobalAveragePooling1D, Bidirectional
from tensorflow.keras import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
from functions import *
from tensorflow import Tensor
from tensorflow.keras.models import Model



def myCreationConv1D_LSTM_B_3(p, filters=32, nLSTM=65, nDense=35):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=43, activation="relu", input_shape=(p.shape[1], 1)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.10))
    model.add(Conv1D(filters=filters, kernel_size=27, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.10))
    model.add(Conv1D(filters=filters, kernel_size=19, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.10))
    model.add(Conv1D(filters=filters, kernel_size=15, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add((Dropout(0.20)))
    model.add(Conv1D(filters=filters, kernel_size=11, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(nLSTM)))
    model.add(Dense(nDense))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer='adam')

    return model




def myCreationConv1D_LSTM1(p, filters=25, kernelSize=25, nLSTM=105):
    model= Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernelSize, activation="relu", input_shape=(1024, 1)))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.30))
    model.add(Bidirectional(LSTM(nLSTM)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model

def myCreationConv1D_LSTM3(p, filters=25, nLSTM=135):
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=23, activation="relu", input_shape=(1024, 1)))
    model.add(Conv1D(filters=64, kernel_size=17, activation="relu"))
    model.add(Conv1D(filters=64, kernel_size=11, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add((Dropout(0.25)))
    model.add(LSTM(nLSTM))
    model.add(Dense(35))
    model.add(Dropout(0.10))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer='adam')

    return model


def myCreationConv1D_LSTM4(p, filters=25, nLSTM=105):
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=43, activation="relu", input_shape=(1024, 1)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.10))
    model.add(Conv1D(filters=64, kernel_size=27, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.10))
    model.add(Conv1D(filters=64, kernel_size=19, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.10))
    model.add(Conv1D(filters=64, kernel_size=11, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add((Dropout(0.20)))
    model.add(LSTM(nLSTM))
    model.add(Dense(45))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer='adam')

    return model

def myCreationConv1D_LSTM_B(p, filters=25, nLSTM=105):
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=43, activation="relu", input_shape=(p.shape[1], 1)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.15))
    model.add(Conv1D(filters=64, kernel_size=27, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.15))
    model.add(Conv1D(filters=64, kernel_size=19, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.15))
    model.add(Conv1D(filters=64, kernel_size=11, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add((Dropout(0.25)))
    model.add(Bidirectional(LSTM(nLSTM)))
    #model.add(Dense(45))
    model.add(Dropout(0.20))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer='adam')

    return model


def myCreationConv1D_LSTM1(p, filters=25, kernelSize=25, nLSTM=105):
    model= Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernelSize, activation="relu", input_shape=(1024, 1)))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.30))
    model.add(Bidirectional(LSTM(nLSTM)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model

def myCreationConv1D_LSTM2(p, filters=25, nLSTM= 15):
    
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=43, activation="relu", input_shape=(1024, 1)))
    model.add(Conv1D(filters=filters, kernel_size=17, activation="relu"))
    model.add(Conv1D(filters=filters, kernel_size=11, activation="relu"))
    model.add(Conv1D(filters=filters, kernel_size=7, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add((Dropout(0.15)))
    model.add(LSTM(nLSTM))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer='adam')

    return model

def myCreationConv1D(p, filters=64, kernel_size=25):
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=43, activation="relu", input_shape=(1024, 1)))
    model.add(Conv1D(filters=filters, kernel_size=25, activation="relu"))
    model.add(Conv1D(filters=filters, kernel_size=17, activation="relu"))
    model.add(Conv1D(filters=filters, kernel_size=7, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add((Dropout(0.15)))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(35))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam') 
    return model

def myCreationConv1D_LSTM_B_2(p, filters=25, nLSTM=105):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=43, activation="relu", input_shape=(p.shape[1], 1)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.10))
    model.add(Conv1D(filters=64, kernel_size=27, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.10))
    model.add(Conv1D(filters=64, kernel_size=19, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.10))
    model.add(Conv1D(filters=64, kernel_size=15, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add((Dropout(0.20)))
    model.add(Conv1D(filters=64, kernel_size=11, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(nLSTM)))
    model.add(Dense(45))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer='adam')

    return model


def myCreationConv1D_LSTM_B_4(p, filters=64, nLSTM=65, nDense=35):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=43, activation="relu", input_shape=(p.shape[1], 1)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.10))
    model.add(Conv1D(filters=filters, kernel_size=27, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.10))
    model.add(Conv1D(filters=filters, kernel_size=19, activation="relu"))
    model.add(Dropout(0.10))
    model.add(Conv1D(filters=filters, kernel_size=15, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add((Dropout(0.20)))
    model.add(Conv1D(filters=filters, kernel_size=11, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=filters, kernel_size=15, activation="relu"))
    model.add(Conv1D(filters=64, kernel_size=17, activation="relu"))
    model.add(Conv1D(filters=64, kernel_size=9, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.15))
    model.add(Bidirectional(LSTM(nLSTM)))
    model.add(Dense(nDense))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer='adam')

    return model

def myCreationConv1D_LSTM_B_5(p, filters=25, nLSTM=105):
        
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=43, activation="relu", input_shape=(p.shape[1], 1)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.10))
    model.add(Conv1D(filters=128, kernel_size=27, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.10))
    model.add(Conv1D(filters=128, kernel_size=19, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.10))
    model.add(Conv1D(filters=128, kernel_size=15, activation="relu"))
    model.add(Conv1D(filters=128, kernel_size=14, activation="relu"))
    model.add(Conv1D(filters=128, kernel_size=13, activation="relu"))
    model.add(Conv1D(filters=128, kernel_size=11, activation="relu"))
    model.add((Dropout(0.10)))
    model.add(Bidirectional(LSTM(nLSTM)))
    model.add(Dense(45))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer='adam')

    return model


def myCreationConv1D_LSTM_B_v2(p, filters=25, nLSTM=150):
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=43, activation="relu", input_shape=(p.shape[1], 1)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.15))
    model.add(Conv1D(filters=64, kernel_size=27, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.15))
    model.add(Conv1D(filters=64, kernel_size=19, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.15))
    model.add(Conv1D(filters=64, kernel_size=11, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add((Dropout(0.25)))
    model.add(Bidirectional(LSTM(nLSTM)))
    model.add(Dense(65))
    model.add(Dense(45))
    model.add(Dense(15))
    model.add(Dropout(0.20))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer='adam')

    return model


def create_res_net(myInputs=(1024, 1)):
        
    inputs = Input(myInputs)
    num_filters = 32
    
    t = BatchNormalization()(inputs)
    t = Conv1D(kernel_size=27,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [1, 3, 3, 1]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        
    
    t = AveragePooling1D(4)(t)
    t = Flatten()(t)
    outputs = Dense(1, activation='relu')(t)
    
    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='mse')

    return model


def create_res_net_LSTM(myInputs=(1024, 1), units_LSTM=135, units_FullyConnected=45, filters=64):
        
    inputs = Input(myInputs)
    num_filters = filters
    
    t = BatchNormalization()(inputs)
    t = Conv1D(kernel_size=27,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 3, 3, 1]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        
    
    t = AveragePooling1D(4)(t)
    #t = Flatten()(t)
    t = Bidirectional(LSTM(units=units_LSTM))(t)
    outputs = Dense(units=units_FullyConnected, activation="relu")(t)
    outputs = Dense(1, activation='relu')(outputs)
    outputs = Dropout(0.1)(outputs)

    model = Model(inputs, outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=opt,
        loss='mse')

    return model


def create_res_net_LSTM_SetD(myInputs=(1024, 1), units_LSTM=135, units_FullyConnected=85, filters=64):
        
    inputs = Input(myInputs)
    num_filters = filters
    
    t = BatchNormalization()(inputs)
    t = Conv1D(kernel_size=27,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 3, 3, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        
    
    t = AveragePooling1D(4)(t)
    t = Bidirectional(LSTM(units=units_LSTM))(t)
    outputs = Dense(units=units_FullyConnected, activation="relu")(t)
    outputs = Dense(2, activation='relu')(outputs)
    

    model = Model(inputs, outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=opt,
        loss='mse')

    return model

# L'architecture qui m'a donné un mse de 1.005.
def create_res_net_SetD(myInputs=(1024, 1), units_LSTM=135, units_FullyConnected=115, filters=64):
        
    inputs = Input(myInputs)
    num_filters = filters
    
    t = BatchNormalization()(inputs)
    t = Conv1D(kernel_size=27,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 3, 3, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        
    
    t = AveragePooling1D(4)(t)
    t = Flatten()(t)
    t = Dropout(0.1)(t)
    #t = Bidirectional(LSTM(units=units_LSTM))(t)
    outputs = Dense(units=units_FullyConnected, activation="relu")(t)
    
    outputs = Dense(2, activation='relu')(outputs)
    

    model = Model(inputs, outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=opt,
        loss='mse')

    return model


# L'achitecture suivante est pour la PPG

def create_res_net_SetD_PPG(myInputs=(1024, 1), units_LSTM = 115, units_FullyConnected=115, filters=64):
        
    inputs = Input(myInputs)
    num_filters = filters
    
    t = BatchNormalization()(inputs)
    t = Conv1D(kernel_size=27,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 3, 3, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        
    
    t = AveragePooling1D(4)(t)
    #t = Flatten()(t)
    t = Dropout(0.2)(t)
    t = Bidirectional(LSTM(units=units_LSTM))(t)
    outputs = Dense(units=units_FullyConnected, activation="relu")(t)
    
    outputs = Dense(2, activation='relu')(outputs)
    

    model = Model(inputs, outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=opt,
        loss='mse')

    return model


# cette architecture à 30 couches est la derniere avec un rmse de 1.90.
def create_res_net_SetD_NewPPG(myInputs=(256, 1), units_LSTM=115, units_FullyConnected=25, filters=64):
        
    inputs = Input(myInputs)
    num_filters = filters
    
    t = BatchNormalization()(inputs)
    t = Conv1D(kernel_size=27,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    t = Dropout(0.1)(t)
    
    num_blocks_list = [3, 4, 4, 3]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        
    
    t = AveragePooling1D(4)(t)
    #t = Flatten()(t)
    
    t = Bidirectional(LSTM(units=units_LSTM))(t)
    #outputs = Dense(units=units_FullyConnected, activation="relu")(t)
    outputs = Dropout(0.2)(t)
    outputs = Dense(2, activation='relu')(outputs)
    

    model = Model(inputs, outputs)
    opt = tf.keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06)
    model.compile(
        optimizer=opt,
        loss=rmse) 

    return model
