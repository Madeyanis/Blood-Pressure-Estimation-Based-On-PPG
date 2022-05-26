# packages
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from functions import diviData
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
import random

# GPU selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Disable eager execution (Adam optimizer cannot be used if this option is enabled)
tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()


# dowlnoad data
pathFolder = '/home/masdoua1u/Desktop/stage/ABP PPG new method/3 ondes NewBDD +/'
pFile = 'PPG3_all.csv'
#tFiler = 'time.csv'
sbpFile = 'SBP3_all.csv'
dbpFiler = 'DBP3_all.csv'
p, SBP, DBP = myDataDownloadFunction(pathFolder, pFile, sbpFile, dbpFiler)


# transform data
p, DBP, SBP = np.array(p), np.array(DBP), np.array(SBP)
#DBP = np.transpose(DBP)
r = random.randint(1, p.shape[0])

# plot d'un échantillon par hazard.
#plt.plot(p[r, :])
#plt.show()


# Normalisation des données
p = myNormalisation(p)
# y = sbp et dbp
y = np.column_stack([SBP, DBP])
# division des données
train_p, test_p, train_y, test_y = diviData(p, y, testSize=0.20)
# expension des dimensions
train_p, test_p = myExpandForNetwork2(train_p, test_p)


#plt.plot(p[r, :])
#plt.show()

# creer un modèle
model = create_res_net_SetD_NewPPG(myInputs=(768, 1))
#model = tf.keras.models.load_model('/home/masdoua1u/Desktop/stage/ABP PPG new method/test 12 BDD sans bousef/best_model.3.853-11.h5')

# ploter l'architecture neuronale en format png
tf.keras.utils.plot_model(
    model,
    to_file="/home/masdoua1u/Desktop/stage/ABP PPG new method/model.png",
    show_shapes=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)



# Callbacks definition 
callbacks_list = [
        #EarlyStopping(monitor='val_loss', patience=1000), # Arret du programme en fonction de la patience souhaitée
        ModelCheckpoint(
        filepath='/home/masdoua1u/Desktop/stage/ABP PPG new method/best_model.{val_loss:.3f}-{epoch:02d}.h5', # sauvegarde du modèle.
        monitor='val_loss', save_best_only=True),
]

EPOCHS = 7000 # nombre d'epoch (itérations)
BATCH_SIZE = 256 # nombre d'échantillons (signaux) pour une mise à jours des poids.
history = model.fit(x=train_p,
                    y=train_y,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_split=0.1,   #  validation data (données servant à améliorer le modèle sans apprentissage.
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks_list, # appel du callback
)

# Sauvegarde de l'historique 
hist_df = pd.DataFrame(history.history)
with open("/home/masdoua1u/Desktop/stage/ABP PPG new method/history_CNN.csv", mode='w') as h:
    hist_df.to_csv(h)

