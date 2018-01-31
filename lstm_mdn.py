import matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sns

import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization, Bidirectional, concatenate, Dense, Dropout, Input, LSTM
# TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

def build_model(windows=3, c=1, m=1):
  # c: the number of outputs we want to predict
  # m: the number of distribution we want to use in the mixture

  def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = K.max(x, axis=axis, keepdims=True)
    return K.log(K.sum(K.exp(x - x_max),
                      axis=axis, keepdims=True)) + x_max

  def mean_log_Gaussian_like(y_true, params):
    """Mean Log Gaussian Likelihood distribution"""
    """Note: The 'c' variable is obtained as global variable"""
    components = K.reshape(params, [-1, c+2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c+1, :]
    alpha = K.softmax(K.clip(alpha, 1e-8, 1.))

    exponent = K.log(alpha) - .5*float(c)*K.log(2*np.pi) \
              - float(c) * K.log(sigma) \
              - K.sum(K.expand_dims((y_true,2)-mu)**2, axis=1)/(2*(sigma)**2)

    log_gauss = log_sum_exp(exponent, axis=1)
    res = -K.mean(log_gauss)
    return res

  def mean_log_LaPlace_like(y_true, params):
    """Mean Log Laplace Likelihood distribution"""
    """Note: The 'c' variable is obtained as global variable"""
    components = K.reshape(params, [-1, c+2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c+1, :]
    alpha = K.softmax(K.clip(alpha, 1e-2, 1.))

    exponent = K.log(alpha) - float(c)*K.log(2*sigma) \
              - K.sum(K.abs(K.expand_dims(y_true,2)-mu), axis=1)/(sigma)

    log_gauss = log_sum_exp(exponent, axis=1)
    res = -K.mean(log_gauss)
    return res

  INPUTS = Input(shape=(1, windows))
  BLSTM1 = Bidirectional(LSTM(7, return_sequences=False))(INPUTS)
  # dropout 
  # lstm ...
  #...
  FC1 = Dense(7)(BLSTM1)
  FC_mus = Dense(c*m)(FC1)
  FC_sigmas = Dense(m, activation=K.exp, kernel_regularizer=l2(1e-3))(FC1)
  FC_alphas = Dense(m, activation='softmax')(FC1)
  OUTPUTS = concatenate([FC_mus, FC_sigmas, FC_alphas], axis=1)
  MODEL = Model(INPUTS, OUTPUTS)

  MODEL.compile(optimizer='adam', loss=mean_log_LaPlace_like)
  print(MODEL.summary())
  return MODEL

def train():
  np.random.seed(12345)
  x = np.linspace(-16, 16, num=300)
  y = np.sin(x)

  print('Shape of x:', str(x.shape), '\nShape of y:', str(y.shape))

  plt.plot(x, y, label='Raw')
  plt.show()

  dataX, dataY = [], []
  window = 3
  for i in range(len(y)-window-1):
    x_ = y[i:(i+window)]
    dataX.append(x_)
    dataY.append(y[i+window])
  dataX, dataY = np.array(dataX), np.array(dataY)
  dataX = dataX.reshape(dataX.shape[0], 1, dataX.shape[1])
  
  model = build_model()
  file_path = "weights_base.best.hdf5"
  checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  earlyStopping = EarlyStopping(monitor="val_loss", mode="min", patience=20)
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, epsilon=0.0001, min_lr=0.0001)
  callbacks = [checkpoint, earlyStopping, reduce_lr]
  history = model.fit(dataX, dataY, epochs=100, batch_size=16, 
                     verbose=2, validation_split=0.1, callbacks=callbacks)

  train_error = model.evaluate(dataX, dataY, verbose=0)
  print('Train Error: ', train_error)

  plt.plot(history.history['loss'])
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['loss'], loc='best')
  plt.show()

  predY = model.predict(dataX)
  xx = x[window:len(x)-1]
  plt.plot(xx, dataY)
  plt.plot(xx, predY[:, 0])
  plt.show()

def test(plot=True):
  model = build_model()
  model.load_weights('weights_base.best.hdf5')

  np.random.seed(12345)
  x = np.linspace(-16, 16, num=300)
  y = np.sin(x)
  dataX, dataY = [], []
  window = 3
  for i in range(len(y)-window-1):
    x_ = y[i:(i+window)]
    dataX.append(x_)
    dataY.append(y[i+window])
  dataX, dataY = np.array(dataX), np.array(dataY)
  dataX = dataX.reshape(dataX.shape[0], 1, dataX.shape[1])

  predY = model.predict(dataX)

  if plot == True:
    pu, pl = predY[:, 0] + predY[:, 1] * 3, predY[:, 0] - predY[:, 1] * 3
    x_ = x[3:len(x)-1]
    plt.plot(x_, y[3:len(x)-1], label='real', c='k')
    plt.plot(x_, predY[:, 0], label='pred', c='r')
    plt.fill_between(x_, pu, pl, facecolor='b', alpha=0.2, linestyle='--')
    plt.legend()
    plt.show()

def main():
  #train()
  test()

if __name__ == '__main__':
  main()


