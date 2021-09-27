#Amina Lemsara 27/09/2021
#MLP prediction for psU rRNA modificationu sing MinION data as training set

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, auc
from tensorflow.keras import regularizers
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.metrics import AUC
from keras import backend as K
print(tf.config.list_physical_devices("GPU"))

X_train = pd.read_pickle("/home/alemsara/DirectRNA/datapsUAll/X_train.pkl")
y_train= np.load("/home/alemsara/DirectRNA/datapsUAll/y_train.npy") 

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# X_test_scaled = scaler.transform(X_test)


def get_keras_model(num_hidden_layers, 
                    dropout_rate, 
                    activation, l1, l2):
    # create the MLP model.
    
    # define the layers.
    inputs = tf.keras.Input(shape=(X_train_scaled.shape[1],))  # input layer.
    x = layers.Dropout(dropout_rate)(inputs) # dropout on the weights.
    
    # Add the hidden layers.
    
    for i in range(num_hidden_layers):
        x = layers.Dense(num_hidden_layers * (num_hidden_layers-i-1)+2, 
                         activation=activation,kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # output layer.
    outputs = layers.Dense(1, activation='sigmoid')(x)
    print(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
    

# This function takes in the hyperparameters and returns a score (Cross validation).
def keras_mlp_cv_score(parameterization,mtr, weight=None):
#     callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=30,min_delta=0)
 
    model = get_keras_model(parameterization.get('num_hidden_layers'),
                            parameterization.get('dropout_rate'),
                            parameterization.get('activation'), 
                            parameterization.get('l1'),
                            parameterization.get('l2'))
    
    opt = parameterization.get('optimizer')
    opt = opt.lower()
    
    learning_rate = parameterization.get('learning_rate')
    
    if opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt == 'rms':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    NUM_EPOCHS = 3000
    loss_fct = parameterization.get('loss')

    # Specify the training configuration.
    model.compile(optimizer=optimizer,
              loss= loss_fct,
              metrics=mtr)

    data = X_train_scaled
    labels = y_train
    class_weights = dict(zip(np.unique(y_train), class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))) 

    # fit the model using a 20% validation set.
    res = model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=parameterization.get('batch_size'),
                    validation_split=0.2, class_weight = class_weights )
    
    # look at the last 10 epochs. Get the mean and standard deviation of the validation score.
#     aucpr_scores = np.array(res.history['val_auc_pr'][-10:])
    pre_scores = np.array(res.history['auc_pr'][-10:])
#     rec_scores = np.array(res.history['val_recall'][-10:])
    
#     precision_scores = np.array(res.history['val_precision'][-10:])
#     recall_scores = np.array(res.history['val_recall'][-10:])

#     mean = 1- (aucpr_scores.mean()+precision_scores.mean()+auc_scores.mean()+recall_scores.mean())/4
    mean = pre_scores.mean()
    sem = pre_scores.std()
    
    # If the model didn't converge then set a high loss.
    if np.isnan(mean):
        return 999.0, 0.0
    
    return mean, sem

parameters=[
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [0.0001, 0.5],
        "log_scale": True,
    },
    {
        "name": "dropout_rate",
        "type": "range",
        "bounds": [0.01, 0.5],
        "log_scale": True,
    },
    {
        "name": "num_hidden_layers",
        "type": "range",
        "bounds": [1, 5],
        "value_type": "int"
    },

    {
        "name": "batch_size",
        "type": "choice",
        "values": [32, 64, 128, 256],
    },
    
    {
        "name": "activation",
        "type": "choice",
        "values": ['tanh', 'relu'],
    },
    {
        "name": "optimizer",
        "type": "choice",
        "values": ['adam', 'rms', 'sgd'],
    },
    {
        "name": "l1",
        "type": "range",
        "bounds": [0.0001, 0.01],
        "log_scale": True,
    },
    {
        "name": "l2",
        "type": "range",
        "bounds": [0.0001, 0.01],
        "log_scale": True,
    },
    {
        "name": "loss",
        "type": "choice",
        "values": ['binary_crossentropy', 'mse'],
    },
]

from ax.service.ax_client import AxClient
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()

ax_client = AxClient()

# create the experiment.
ax_client.create_experiment(
    name="keras_experiment",
    parameters=parameters,
    objective_name='keras_cv',
    minimize=False)
mtr = AUC(curve='PR', name='auc_pr')

def evaluate(parameters):
    return {"keras_cv": keras_mlp_cv_score(parameters, mtr)}
for i in range(100):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
    ax_client.get_trials_data_frame().sort_values('trial_index')
best_parameters, values = ax_client.get_best_parameters()

# the best set of parameters.
print('thebestparameters are:')
for k in best_parameters.items():
  print(k)

print()

# the best score achieved.
means, covariances = values
print(means)
# keras_model = get_keras_model(best_parameters['num_hidden_layers'], 
#                               best_parameters['dropout_rate'],
#                               best_parameters['activation'],                            
#                               best_parameters['l1'],
#                               best_parameters['l2'])

# opt = best_parameters['optimizer']
# opt = opt.lower()

# learning_rate = best_parameters['learning_rate']

# if opt == 'adam':
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# elif opt == 'rms':
#     optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
# else:
#     optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# NUM_EPOCHS = 1000

# loss_fct = best_parameters.get('loss')
# # Specify the training configuration.
# keras_model.compile(optimizer=optimizer,
#               loss=loss_fct,
#               metrics=['AUC'])

# data = X_train_scaled
# labels = y_train
# class_weights = dict(zip(np.unique(y_train), class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))) 
# res = keras_model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=best_parameters['batch_size'], class_weight = class_weights)

# test_pred = keras_model.predict(X_test_scaled)

# np.save("test_pred_Y_0", test_pred)
