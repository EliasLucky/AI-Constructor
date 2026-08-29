from django.conf import settings

import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from tensorflow.keras.callbacks import Callback

from tensorflow.keras import backend as K

from celery.result import AsyncResult

import joblib
import pandas as pd
import numpy as np

import os

import time

path = os.path.join(settings.BASE_DIR, 'api', 'exoplanet_model.h5')
active_model = tf.keras.models.load_model(path)
default_model = active_model
scaler = joblib.load(os.path.join(settings.BASE_DIR, 'api', 'scaler.pkl'))
features = ['pl_orbper', 'pl_trandur', 'pl_rade', 'pl_bmasse', 
            'st_teff', 'st_rad', 'st_mass', 'sy_dist']

class Monitor(Callback):
    def __init__(self, task_id):
        super().__init__()
        self.task_id = task_id
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        #AsyncResult(self.task_id).update_state(
        self.task_id.update_state(
            state='PROGRESS',
            meta={
                'status': 'PROGRESS',
                'progress': {
                    'current_epoch': epoch+1,
                    'total_epochs': self.params['epochs'],
                    'training_time': time.time() - self.start_time
                }
            }
        )

def recall(y_true, y_pred):
    num_classes = 3
    y_true = tf.cast(y_true ,tf.int32)
    y_pred_labels = tf.argmax(y_pred, axis=1,output_type=tf.int32)
    confusion_matrix = tf.confusion_matrix(y_true,y_pred_labels,num_classes=num_classes)
    diag = tf.linalg.diag_part(confusion_matrix)
    row_sums = tf.reduce_sum(confusion_matrix,axis=1)
    recall_per_class = diag / (row_sums + tf.keras.backend.epsilon())
    return tf.reduce_mean(recall_per_class)

def precision(y_true,y_pred):
    num_classes = 3
    y_true = tf.cast(y_true,tf.int32)
    y_pred_labels = tf.argmax(y_pred, axis=1, output_type=tf.int32)
    confusion_matrix = tf.confusion_matrix(y_true,y_pred_labels,num_classes=num_classes)
    diag = tf.linalg.diag_part(confusion_matrix)
    col_sums = tf.reduce_sum(confusion_matrix,axis=0)
    precision_per_class = diag / (col_sums + tf.keras.backend.epsilon())
    return tf.reduce_mean(precision_per_class)

def predict_exoplanets(df, model):
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    df_clean = df.dropna(subset=features)
    X = df_clean[features]
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    pred_labels = np.argmax(predictions, axis=1)
    label_map = {0: 'CONFIRMED', 1: 'CANDIDATE', 2: 'FALSE POSITIVE'}
    
    df_clean['prediction'] = [label_map[p] for p in pred_labels]
    return df_clean

def recall_m(y_true,y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true,0,1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true,y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred,0,1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred,0,1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def retrain_model_with_data(df,hyperparams,architecture,id,max_epochs):
    target = 'tfopwg_disp'

    df_clean = df.dropna(subset=features+[target])
    X = df_clean[features]
    y = df_clean[target].map({'CONFIRMED': 0,'CANDIDATE':1,'FALSE POSITIVE':2})

    model = build_model_from_architecture(architecture,hyperparams)

    if hyperparams['use_batchnorm']:
        model.add(BatchNormalization())

    #if hyperparams['l2_reg']:
    #    model.add(tf.keras.regularizers.l2(0.001))

    optimizer = hyperparams.get('optimizer', 'adam')
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=hyperparams['learning_rate'],beta_1=hyperparams['beta_1'],beta_2=hyperparams['beta_2'])
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=hyperparams['learning_rate'])
    else:
        optimizer = SGD(learning_rate=hyperparams['learning_rate'], momentum=hyperparams.get('momentum',0.9),nesterov=hyperparams.get('nesterov', False))

    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    callback = None

    monitor = Monitor(id)
    history = model.fit(X,y,epochs=hyperparams['epochs'],batch_size=hyperparams['batch_size'],validation_split=hyperparams.get('val_split', 0.2),callbacks=[monitor],verbose=0)
    
    val_loss,val_accuracy = model.evaluate(X,y,verbose=0)
    metrics = {
        "loss": val_loss,"accuracy": val_accuracy
    }
    
    
    return history,metrics,model,monitor

def build_model_from_architecture(architecture,hyperparams):
    model = Sequential()

    for i, layer in enumerate(architecture):
        if layer['type'] == 'input':
            continue

        elif layer['type'] == 'dense':
            if i == 1:
                model.add(Dense(layer['units'],activation=layer['activation'],input_shape=(8,)))
            else:
                model.add(Dense(layer['units'],activation=layer['activation'],kernel_regularizer=tf.keras.regularizers.l2(hyperparams['l2_reg']) if hyperparams['use_reg'] else None))

            #if layer['dropout']:
            #    model.add(Dropout(layer['rate']))

            #if layer['batch_norm']:
            #    model.add(BatchNormalization())

        elif layer['type'] == 'dropout':
            model.add(Dropout(layer['dropout']))

        elif layer['type'] == 'batchnorm':
            model.add(BatchNormalization())

        elif layer['type'] == 'output':
            model.add(Dense(layer['units'],activation=layer['activation']))

    return model

def preprocess_data(df):
    target = 'disposition'
    df_clean = df.dropna(subset=features+[target])
    X = df_clean[features]
    y = df_clean[target].map({'CONFIRMED':0,'CANDIDATE':1,'FALSE POSITIVE':2})

    return X,y