import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import math

class Histories(keras.callbacks.Callback):

  def on_train_begin(self, logs={}):
    self.loss = []
    self.val_loss = []

  def on_train_end(self, logs={}):
    return

  def on_epoch_begin(self, epoch, logs={}):
    return

  def on_epoch_end(self, epoch, logs={}):
    self.loss.append(logs.get('loss'))
    self.val_loss.append(logs.get('val_loss'))
    return

  def on_batch_begin(self, batch, logs={}):
    return

  def on_batch_end(self, batch, logs={}):
    return
