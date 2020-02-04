# This file contains all custom elements for the NN model.
#
# Note that when the model is saved, these functions are NOT saved,
# so they must be loaded explicitly when restoring the model from file!

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  #to catch FutureWarnings
    import tensorflow.keras.backend as K  # needed for custom loss function
    import tensorflow as tf

################################################################
# Custom metric
def my_r_square_metric(y_true,y_pred):
   ss_res = K.sum(K.square(y_true-y_pred))
   ss_tot = K.sum(K.square(y_true-K.mean(y_true)))
   return ( 1 - ss_res/(ss_tot + K.epsilon()) )
   
################################################################

def my_mean_squared_error_noweight(y_true,y_pred):
    return K.mean( tf.square(tf.subtract(y_pred,y_true)) )

def my_mean_squared_error_weighted1(y_true,y_pred):
    return K.mean( tf.multiply( tf.exp(tf.multiply(5.0,y_true)) , tf.square(tf.subtract(y_pred,y_true)) ) )

def my_mean_squared_error_weighted(weight=0.0):
    def loss(y_true,y_pred):
        return K.mean( tf.multiply( tf.exp(tf.multiply(weight,y_true)) , tf.square(tf.subtract(y_pred,y_true)) ) )
    return loss

################################################################
