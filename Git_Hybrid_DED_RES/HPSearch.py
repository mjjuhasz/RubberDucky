
# Regression
# %tensorflow_version 2.x
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.keras.backend.clear_session()  # For easy reset of notebook state.
# Load the TensorBoard notebook extension
# %load_ext tensorboard
# Clear any logs from previous runs
# !rm -rf ./logs/

# load the dataset

import pandas as pd

df = pd.read_excel('~/Documents/Python/Hybrid_DED_RES/PH138MLData.xlsx', sheet_name='Data')

# Set mean and standard deviation
mean_std={}
for var in df.columns:
    mean_std[var]=(df[var].mean(), df[var].std())

# Standardize ranges
from sklearn import preprocessing
preprocessing.scale
df_z_score=df.apply(preprocessing.scale)

# split into input (X) and output (y) variables
x = df_z_score.loc[::,['Linear Heat Input (J/mm)', 'Mass Flow (g/min)']]
y = df_z_score.loc[::,['Aspect Ratio', '% of Nominal Spot Size']]

# Create train/test

from sklearn.model_selection import cross_val_score, KFold, train_test_split

x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.1, random_state=21)

# Build neural network
def Regressor(Neurons,LearningRate,NumLayers):
  # Set up the model
  regRL=tf.keras.regularizers.L1L2(l1=0.0, l2=.05)
  # Create Input Layer
  inputs = tf.keras.Input(shape=(2,))
  x = layers.Dense(Neurons,activation='relu')(inputs)
  # Hidden Layers
  for n in range(0,NumLayers):
    x = layers.BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
    x = layers.Dense(Neurons,activation='relu',use_bias=True, kernel_regularizer=regRL)(x)
  # Create Output Layer
  outputs = layers.Dense(2)(x)
  model=tf.keras.Model(inputs=inputs,outputs=outputs)
  # Compile Model and setup parameters
  optimizer=tf.keras.optimizers.Nadam(learning_rate=LearningRate)
  loss=tf.keras.losses.MeanSquaredError()
  #loss=tf.keras.losses.KLDivergence()
  model.compile(optimizer,loss,metrics=['acc'])
  return model

SkReg = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=Regressor, epochs=20, batch_size=5, verbose=0)

import numpy as np
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV

# define the grid search parameters
Neurons = [1500,2000,2500]
LearningRate = [1e-4]
NumLayers = [2]
# Ridge = stats.uniform(0,1)
# Lasso = stats.uniform(0,1)

Space = dict(Neurons=Neurons, LearningRate=LearningRate, NumLayers=NumLayers)

HPSearch = GridSearchCV(estimator=SkReg, param_grid=Space, n_jobs=-1,verbose=1)

HP_results = HPSearch.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (HP_results.best_score_, HP_results.best_params_))
means = HP_results.cv_results_['mean_test_score']
stds = HP_results.cv_results_['std_test_score']
params = HP_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
