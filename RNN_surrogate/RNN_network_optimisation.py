# -*- coding: utf-8 -*-
"""
Desc: script for training RNN surrogate model
Created on 01/08/2023 12:07
@author: shrro
"""

# mlp for multi-output regression
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Add
import keras.backend as K
from keras.losses import Huber
import numpy as np
import scipy.io
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt
import joblib
from tensorflow.keras.layers.experimental import preprocessing


def residual_block(x, block_depth=2, hidden_size=30, dropout_rate=0.0):
    # store the input tensor to be added later as the identity
    identity = x

    for _ in range(block_depth):
        x = Dense(hidden_size, kernel_initializer='he_uniform', activation='swish')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    x = Add()([identity, x])

    return x

class sr_RESNET(kt.HyperModel):

    def __init__(self, normalizer, output_size, input_size, Bs, Bv, ns):
        self.normalizer = normalizer
        self.output_size = output_size
        self.input_size = input_size
        self.Bs = Bs
        self.Bv = Bv
        self.ns = ns

    def build(self, hp):
        inputs = keras.Input(shape=(self.input_size,))
        x1 = self.normalizer(inputs)

        hs = hp.Int('hidden_size', 10, 120, step=15, default=40)
        do = hp.Float('dropout_rate', 0.0, 0.4, step=0.1)

        x1 = Dense(hs, kernel_initializer='he_uniform', activation='swish')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(do)(x1)

        nb = hp.Int('n_blocks', 1, 4, default=2)
        bd = hp.Int('block_depth', 1, 4, default=2)

        for _ in range(nb):
            x1 = residual_block(x1, block_depth=bd, hidden_size=hs, dropout_rate=do)

        outputs = Dense(self.output_size)(x1)

        lr = hp.Float('learning_rate', 1e-2, 5e-1, sampling='log')

        model = keras.Model(inputs=inputs, outputs=outputs, name="resnet_v1")
        model.compile(
            optimizer=keras.optimizers.Adagrad(
                learning_rate=lr),
            loss=matrix_loss(Bs=self.Bs, Bv=self.Bv, ns=self.ns))

        return model

def matrix_loss(Bs,Bv,ns):
    def loss(y_true, y_pred):
        pySEC = K.dot(y_true[:,0:ns], Bs)
        pyVEL = K.dot(y_true[:,ns:],Bv)
        uaSEC = K.dot(y_pred[:,0:ns], Bs)
        uaVEL = K.dot(y_pred[:,ns:],Bv)
        mse  =Huber()
        loss1 = mse(uaSEC,pySEC)
        loss2 = mse(uaVEL,pyVEL)
        return loss1 + loss2
    return loss

# load dataset
mat = scipy.io.loadmat('C:/Users/shrro/PycharmProjects/RNN_surrogate/mat_files/data_N0k90.mat')
mat2 = scipy.io.loadmat('C:/Users/shrro/PycharmProjects/RNN_surrogate/mat_files/SVD_N0k90.mat')

X_train = mat['X_train']
Y_train = mat['T_train']

X_val = mat['X_val']
Y_val = mat['T_val']

X_test = mat['X_test']
Y_test = mat['T_test']

num_outputs = len(Y_train[0])
num_inputs = len(X_train[0])

Bs = mat2['Bs']
Bv = mat2['Bv']
sec_pct = mat2['sec_pct']
ns = np.shape(sec_pct)[0]

Bv2 = tf.convert_to_tensor(Bv, dtype=tf.float32)
Bs2 = tf.convert_to_tensor(Bs, dtype=tf.float32)

scaler = StandardScaler()
scaler.fit(Y_train)

trainy_scaled = scaler.transform(Y_train)
valy_scaled = scaler.transform(Y_val)
testy_scaled = scaler.transform(Y_test)

normalizer = preprocessing.Normalization()
normalizer.adapt(X_train)

tuner = kt.BayesianOptimization(
    sr_RESNET(normalizer, num_outputs, num_inputs, Bs2, Bv2, ns),
    objective='val_loss',
    max_trials=120,
    max_consecutive_failed_trials=2,
    directory="/tmp/tb7",
    overwrite=False
)

tuner.search(x=X_train, y=trainy_scaled,
                    validation_data=(X_val, valy_scaled),
                    epochs=2000, batch_size=32, shuffle=True, verbose=1,
                    callbacks=[
                        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50),
#                        keras.callbacks.TensorBoard("/tmp/tb7_logs")
                    ])


best_model = tuner.get_best_models(1)[0]

best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
print(best_hyperparameters.values)

# save model
best_model.save("tuned_model_forAll_N90.keras")
joblib.dump(scaler, 'tuned_scaler_forAll_N90.joblib')