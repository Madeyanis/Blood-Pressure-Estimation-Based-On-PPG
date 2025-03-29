# packages
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import pandas as pd
import numpy as np
from functions import *
from ModelsDefinitions import *
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv1D, ReLU, BatchNormalization,\
                                    Add, AveragePooling1D, Flatten, Dense
from tensorflow.keras.models import Model
import datetime



# GPU selection
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Disable eager execution (Adam optimizer cannot be used if this option is enabled)
tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()


model = tf.keras.models.load_model('best_model.1.407-02.h5')

pathFolder = '/home/masdoua1u/Desktop/stage/Data/'
pFile = 'PPG3_all.csv'
#tFiler = 'time.csv'
sbpFile = 'SBP3_all.csv'
dbpFiler = 'DBP3_all.csv'
p, t, SBP, DBP = myDataDownloadFunction(pathFolder, pFile, tFiler, sbpFile, dbpFiler)

# transform data
p, t, DBP, SBP = np.array(p), np.array(t), np.array(DBP), np.array(SBP)

# Normalisation des données
p = myNormalisation(p)

# splitt data
train_p, test_p, train_SBP, test_SBP = diviData(p, SBP, testSize=0.15)
y = np.column_stack([SBP, DBP])

# Expend dimensions
train_p, test_p = myExpandForNetwork2(train_p, test_p)

p = myExpandForNetwork(p)


myEvaluation(model, p, y)

