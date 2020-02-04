# BEFORE RUNNING THIS FUNCTION:
# Must have executed function MAIN_PREPARE_SAVE_DATA to generate data files.

import pickle
from datetime import datetime
import sys

# Self-defined functions
from load_data import load_data
from custom_model_elements import my_r_square_metric
from read_configuration import read_configuration
from default_configuration import defcon
from make_custom_file_names import model_file_name
from make_custom_file_names import history_file_name
from make_custom_file_names import data_file_name

# custom loss functions
from custom_model_elements import my_mean_squared_error_noweight
from custom_model_elements import my_mean_squared_error_weighted1
# Note: also need to set this function below the line "### LOSS FUNCTION"

import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.simplefilter(action='ignore',category=DeprecationWarning)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# optional command line argument: configuration file name
try:
    config_file_name = sys.argv[1]
    config = read_configuration(config_file_name)
except IndexError:
    config = read_configuration()

################################################################

# machine specific configuration

spath = '..'  #Imme
verbose_fit = 1

try:
    machine = config['machine']
except KeyError:
    machine = defcon['machine']
print('machine =',machine)

if machine == 'Hera':

    # to avoid GPU out of memory error
    cp = tf.ConfigProto()
    cp.gpu_options.allow_growth = True
    session = tf.Session(config=cp)

    spath = '/scratch1/RDARCH/rda-goesstf/conus2'  #KH on Hera
    verbose_fit = 2  #one progress bar per epoch

################################################################

print('start MAIN_TRAIN_and_SAVE_MODEL=',datetime.now())

##### Load data
try:
    data_suffix = config['data_suffix']
except KeyError:
    data_suffix = defcon['data_suffix']
print('data_suffix =',data_suffix)

data_file = data_file_name( spath, suffix=data_suffix ) # load file name from file
print('loading data from file =',data_file)
Xdata_train, Ydata_train, Xdata_test, Ydata_test, \
    Lat_train, Lon_train, Lat_test, Lon_test = load_data( data_file )
nbatches_train,ny,nx,nchans = Xdata_train.shape

################################################################

##### Load configuration
# Use prefix to add custom names to the file names generated here,
# to make sure we don't overwrite model files, etc.
# Example:  'I1', where I is for Imme and 1 denotes Experiment #1.
print('Configuration:')
try:
    my_file_prefix = config['my_file_prefix']
except KeyError:
    sys.exit('Error: you must supply my_file_prefix in configuration file')
print('my_file_prefix =',my_file_prefix)

### parameter choices for training ###

try:
    NN_string = config['NN_string']
except KeyError:
    NN_string = defcon['NN_string']
print('NN_string =',NN_string)

try:
    activ = config['activ']
except KeyError:
    activ = defcon['activ']
print('activ =',activ)

try:
    activ_last = config['activ_last']
except KeyError:
    activ_last = defcon['activ_last']
print('activ_last =',activ_last)

try:
    batch_size = config['batch_size']
except KeyError:
    batch_size = int(nbatches_train/defcon['batch_step_size'])
print('batch_size =',batch_size)

try:
    batchnorm = config['batchnorm']
except KeyError:
    batchnorm = defcon['batchnorm']
print('batchnorm=',batchnorm)

try:
    convfilter = config['convfilter']
except KeyError:
    convfilter = defcon['convfilter']
print('convfilter =',convfilter)

try:
    convfilter_last_layer = config['convfilter_last_layer']
except KeyError:
    convfilter_last_layer = defcon['convfilter_last_layer']
print('convfilter_last_layer =',convfilter_last_layer)

try:
    double_filters = config['double_filters']
except KeyError:
    double_filters = defcon['double_filters']
print('double_filters =',double_filters)

try:
    dropout = config['dropout']
except KeyError:
    dropout = defcon['dropout']
print('dropout = ',dropout)

if dropout:
    try:
        dropout_rate = config['dropout_rate']
    except KeyError:
        dropout_rate = defcon['dropout_rate']
    print('dropout_rate = ',dropout_rate)

try:
    kernel_init = config['kernel_init']
except KeyError:
    kernel_init = defcon['kernel_init']
print('kernel_init =',kernel_init)

### LOSS FUNCTION
try:
    loss = config['loss']
except KeyError:
    loss = defcon['loss']
print('loss =',loss)
if loss == 'my_mean_squared_error_noweight': loss = my_mean_squared_error_noweight
if loss == 'my_mean_squared_error_weighted1': loss = my_mean_squared_error_weighted1
###

try:
    n_conv_layers_per_decoder_layer = \
        config['n_conv_layers_per_decoder_layer']
except KeyError:
    n_conv_layers_per_decoder_layer = \
        defcon['n_conv_layers_per_decoder_layer']
print('n_conv_layers_per_decoder_layer =',n_conv_layers_per_decoder_layer)

try:
    n_conv_layers_per_encoder_layer = \
        config['n_conv_layers_per_encoder_layer']
except KeyError:
    n_conv_layers_per_encoder_layer = \
        defcon['n_conv_layers_per_encoder_layer']
print('n_conv_layers_per_encoder_layer =',n_conv_layers_per_encoder_layer)

try:
    n_encoder_decoder_layers = config['n_encoder_decoder_layers']
except KeyError:
    n_encoder_decoder_layers = defcon['n_encoder_decoder_layers']
print('n_encoder_decoder_layers =',n_encoder_decoder_layers)

try:
    n_filters_for_first_layer = config['n_filters_for_first_layer']
except KeyError:
    n_filters_for_first_layer = defcon['n_filters_for_first_layer']
print('n_filters_for_first_layer =',n_filters_for_first_layer)

try:
    n_filters_last_layer = config['n_filters_last_layer']
except KeyError:
    n_filters_last_layer = defcon['n_filters_last_layer']
print('n_filters_last_layer =',n_filters_last_layer)

try:
    nepochs = config['nepochs']
except KeyError:
    nepochs = defcon['nepochs']
print('nepochs =',nepochs)

try:
    poolfilter = config['poolfilter']
except KeyError:
    poolfilter = defcon['poolfilter']
print('poolfilter =',poolfilter)

try:
    upfilter = config['upfilter']
except KeyError:
    upfilter = defcon['upfilter']
print('upfilter =',upfilter)

##### part below does not change

if NN_string == 'SEQ':
    IS_UNET = False
else:
    IS_UNET = True
print('IS_UNET =',IS_UNET)

layer_format = ['P','CP','CCP','CCCP','CCCCP']
print('encoder layer_format =',layer_format[n_conv_layers_per_encoder_layer])
print('decoder layer_format =',layer_format[n_conv_layers_per_decoder_layer])

optimizer = Adam()
print('optimizer = Adam')

padding = 'same'
print('padding = ',padding)

metrics = [my_r_square_metric,'mean_absolute_error']
if loss != 'mean_squared_error':
    metrics.append('mean_squared_error')

# Tell the user about architecture
#if IS_UNET:
#    print('\nArchitecture: Unet')
#else:
#    print('\nArchitecture: Standard sequential')
#print('Blocks: ' + repr(n_encoder_decoder_layers))
#print('Epochs: ' + repr(nepochs))
#print('Batch size: ' + repr(batch_size))

##### Get file names
modelfile = model_file_name(spath, IS_UNET, my_file_prefix, \
    n_encoder_decoder_layers, nepochs )
historyfile = history_file_name( spath, IS_UNET, my_file_prefix, \
    n_encoder_decoder_layers, nepochs )

print('\nResults will be stored here:')
print( '   Model file: ' + modelfile )
print( '   History file: ' + historyfile )

################################################################
# Define model: encoder-decoder structure

stime = datetime.now()
print('\nstart',stime)

n_filters = n_filters_for_first_layer

if IS_UNET:
    ##### Define Unet
    input = Input(shape=(ny, nx, nchans))
    skip = []
    x = input

    for i_encode_decoder_layer in range(n_encoder_decoder_layers):
        skip.append(x)  # push current x on top of stack
        for i in range(n_conv_layers_per_encoder_layer): #add conv layer
            x = Conv2D(n_filters,convfilter,activation=activ,\
                padding=padding,kernel_initializer=kernel_init)(x)
            if batchnorm:
                x = BatchNormalization()(x)
        x = MaxPooling2D(poolfilter, padding=padding)(x)
        if dropout:
            x = Dropout(dropout_rate)(x)
        if double_filters:
            n_filters = n_filters * 2 # double for NEXT layer

    for i_encode_decoder_layer in range(n_encoder_decoder_layers):
        for i in range(n_conv_layers_per_decoder_layer): #add conv layer
            x = Conv2DTranspose(n_filters,convfilter,activation=activ,\
                padding=padding,kernel_initializer=kernel_init)(x)
            if batchnorm:
                x = BatchNormalization()(x)
        x = UpSampling2D(upfilter)(x)
        x = Concatenate()([x,skip.pop()]) # pop top element
        if dropout:
            x = Dropout(dropout_rate)(x)
        if double_filters:
            n_filters = n_filters // 2 # halve for NEXT layer

    # One additional (3x3) conv layer to properly incorporate newly
    # added channels at previous concatenate step
    x = Conv2D(n_filters,convfilter,activation=activ,\
        padding=padding,kernel_initializer=kernel_init)(x)

    # last layer: 2D convolution with (1x1) just to merge the channels
    x = Conv2D(n_filters_last_layer,convfilter_last_layer,\
        activation=activ_last,padding=padding,\
        kernel_initializer=kernel_init)(x)
    x = Reshape((ny,nx))(x)
    model = Model(inputs=input, outputs=x)
    print('Unet !!!!')

else:
    ##### Define standard encoder-decoder network (no skip connections)
    model = Sequential()

    ### contracting path (encoder layers)
    for i_encode_decoder_layer in range( n_encoder_decoder_layers ):
        print('Add encoder layer #' + repr(i_encode_decoder_layer) )

        for i in range(n_conv_layers_per_encoder_layer): #add conv layer
            model.add(Conv2D(n_filters,convfilter,activation=activ,\
                padding=padding,input_shape=(ny,nx,nchans), \
                kernel_initializer=kernel_init))
            if batchnorm:
                model.add(BatchNormalization())
        model.add(MaxPooling2D(poolfilter,padding=padding))
        if dropout:
            model.add(Dropout(dropout_rate))
        if double_filters:
            n_filters = n_filters * 2 # double for NEXT layer

    ### expanding path (decoder layers)
    for i_encode_decoder_layer in range( n_encoder_decoder_layers ):
        print('Add decoder layer #' + repr(i_encode_decoder_layer) )

        for i in range(n_conv_layers_per_decoder_layer): #add conv layer
            model.add(Conv2DTranspose(n_filters,convfilter,\
                activation=activ,padding=padding,\
                kernel_initializer=kernel_init))
            if batchnorm:
                model.add(BatchNormalization())
        model.add(UpSampling2D(upfilter))
        if dropout:
            model.add(Dropout(dropout_rate))
        if double_filters:
            n_filters = n_filters // 2 # halve for NEXT layer

    # last layer: 2D convolution with (1x1) just to merge the channels
    model.add(Conv2D(n_filters_last_layer,convfilter_last_layer,\
        activation=activ_last,padding=padding,\
        kernel_initializer=kernel_init))
    model.add(Reshape((ny,nx)))

# Architecture definition complete

print('\n')
print(model.summary())  # print model architecture

#    print('# encoder/decoder layers = ' + repr(n_encoder_decoder_layers))
#    print('# epochs =' + repr(nepochs))
#    print('batch size =' + repr(batch_size))

########### TRAIN MODEL ###########
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

history = model.fit(Xdata_train,Ydata_train,epochs=nepochs,\
    batch_size=batch_size,shuffle=True,\
    validation_data=(Xdata_test,Ydata_test), \
    verbose=verbose_fit)

# Time statistics
etime = datetime.now()
print('end',etime)
print('Time ellapsed for training',(etime-stime).total_seconds(),\
    'seconds\n')

########### SAVE MODEL ###########
print('Writing model to file: ' + modelfile + '\n')
model.save(modelfile)

########### Save training history ###########
print('Writing history to file: ' + historyfile + '\n')
with open(historyfile, 'wb') as f:
    pickle.dump({'history':history.history, 'epoch':history.epoch}, f)

######################################################################

print('TRAINING:  Done!')
print('To inspect results, run function '\
    'MAIN_POST_PROCESSING or function MAIN_VISUALIZATION\n')

print('end MAIN_TRAIN_and_SAVE_MODEL=',datetime.now())
