import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import ReLU, BatchNormalization, Add, Activation, Conv1D, Dropout, MaxPooling1D, Flatten, Dense, LSTM, Reshape, GlobalAveragePooling1D, Bidirectional
from tensorflow.keras import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import Tensor
from keras import backend


# diviser les données en train_set et test_set
def diviData(x, y, testSize=0.25):

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = testSize)

    return train_x, test_x, train_y, test_y


# cette fonction est optionnelle.
def testerUnSignal(model, s, batchSize=128):

    s = np.expand_dims(s, axis=0)
    r = model.predict(s, batch_size=batchSize)

    return r

# fonction optionnelle.
def testerUnSignalAvecPrint(model, s, y, batchSize=1024):

    s = np.expand_dims(s, axis=0)
    r = model.predict(s, batch_size=batchSize)
    print("value predicted = {} and the reel value is = {}\n mse = {}".format(r, y, abs(r - y)))
    

# Normalisation par moyenne et écart-type
def myNormalisation(p):

    if len(p.shape) == 1:
        p = p - np.mean(p)
        p = p / np.std(p) 
    else: 
        for i in range(p.shape[0]):
            
            p[i] = p[i] - np.mean(p[i])
            p[i] = p[i] / np.std(p[i])
    
    return p


# fonction pour évaluer un modèle
def myEvaluation(model, test_p, test_SBP, batchSize=256):
    
    print("evaluating the model, test loss: {}".format(model.evaluate(test_p, test_SBP,batch_size=batchSize)))

# expend un tensor nécessaire pour le réseau pour 2 vecteurs
def myExpandForNetwork2(train_p, test_p):
    train_p = np.expand_dims(train_p, axis=2)
    test_p = np.expand_dims(test_p, axis=2)

    return train_p, test_p

# expend pour un seul vecteur
def myExpandForNetwork(p):
    p = np.expand_dims(p, axis=2)

    return p

# download des données csv ou excel (dans ce cas csv).
def myDataDownloadFunction(pathFolder, pFile, sbpFile, dbpFile):

    p = pd.read_csv((pathFolder + pFile), header=None)
    #t = pd.read_csv((pathFolder + tFile), header=None)
    DBP = pd.read_csv((pathFolder + dbpFile), header=None)
    SBP = pd.read_csv((pathFolder + sbpFile), header=None)

    return p, SBP, DBP

# fonction pour ploter l'historique pour le learning cave.
def myHistoryPlot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation' ], loc='upper left')
    plt.show()

# J'ai oublié pourquoi j'ai crée une deuxième fonction plot.
def myHistoryPlot2(history):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation' ], loc='upper left')
    plt.show()

# les résiduels bloc pour concevoir une architecture résiduelle
def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 23) -> Tensor:
    y = Conv1D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Dropout(0.1)(y)
    y = Conv1D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)
    y = Dropout(0.1)(y)

    if downsample:
        x = Conv1D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = Dropout(0.1)(out)
    
    out = relu_bn(out)
    return out 


# fonction d'activation
def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

# normalisation par max et min.
def myNormalisationZeroUn(p):
    
    if len(p.shape) == 1:
        p = p - np.min(p)
        p = p / np.max(p) 
    else: 
        for i in range(p.shape[0]):
            
            p[i] = p[i] - np.min(p[i])
            p[i] = p[i] / np.max(p[i])
    
    return p

# la fonction rmse adaptée pour l'optimiseur dans l'architecture du ResNet 
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# rmse en utilisant numpy (non adaptée pour l'optimiseur.
def rmseNumpy(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())
