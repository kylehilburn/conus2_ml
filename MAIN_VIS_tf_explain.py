import numpy as np
from matplotlib import pyplot as plt
import cv2
import sys

### Load methods to use from tf-explain
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.smoothgrad import SmoothGrad

# Self-defined functions
from load_data import load_data
from make_custom_file_names import model_file_name
from make_custom_file_names import history_file_name
from make_custom_file_names import data_file_name
from make_custom_file_names import heat_map_file_name_start
from custom_model_elements import my_r_square_metric
from prepare_data import ymax_default as ymax
from read_configuration import read_configuration
from default_configuration import defcon

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  #to catch FutureWarnings
    import tensorflow as tf
    if not tf.executing_eagerly():
        tf.enable_eager_execution()
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Flatten

# required command line argument: my_file_prefix
try:
    my_file_prefix = sys.argv[1]
except IndexError:
    sys.exit('Error: you must supply my_file_prefix as command line argument')
print('my_file_prefix =',my_file_prefix)

config = read_configuration()

##############################
### Choose model for which to run the visualization - by parameters:

spath = '..'  #Imme

try:
    machine = config['machine']
except KeyError:
    try:
        machine = config[my_file_prefix]['machine']
    except KeyError:
        machine = defcon['machine']
print('machine =',machine)

if machine == 'Hera':
    spath = '/scratch1/RDARCH/rda-goesstf/conus2'  #KH on Hera

try:
    data_suffix = config['data_suffix']
except KeyError:
    try:
        data_suffix = config[my_file_prefix]['data_suffix']
    except KeyError:
        data_suffix = defcon['data_suffix']
print('data_suffix =',data_suffix)

try:
    NN_string = config['NN_string']
except KeyError:
    try:
        NN_string = config[my_file_prefix]['NN_string']
    except KeyError:
        NN_string = defcon['NN_string']
print('NN_string =',NN_string)

if NN_string == 'SEQ':
    IS_UNET = False
else:
    IS_UNET = True
print('IS_UNET =',IS_UNET)

try:
    n_encoder_decoder_layers = config['n_encoder_decoder_layers']
except KeyError:
    try:
        n_encoder_decoder_layers = config[my_file_prefix]['n_encoder_decoder_layers']
    except KeyError:
        n_encoder_decoder_layers = defcon['n_encoder_decoder_layers']
print('n_encoder_decoder_layers =',n_encoder_decoder_layers)

try:
    nepochs = config['nepochs']
except KeyError:
    try:
        nepochs = config[my_file_prefix]['nepochs']
    except KeyError:
        nepochs = defcon['nepochs']
print('nepochs =',nepochs)

ALSO_PREDICT_TRAINING_DATA = False
#WANT_CONVERGENCE_PLOT = False
#WANT_PREDICTION_PLOTS = False
#WANT_FEATURE_MAPS = True
#WANT_LRP_PLOTS = True

######################################################################
### Filenames for model, history, data, etc.
modelfile = model_file_name(\
    spath, IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs )
historyfile = history_file_name( \
    spath, IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs )
data_file = data_file_name( spath, suffix=data_suffix ) # load file name from file

######################################################################
################# DATA AND ANN MODEL ##########
# Step 1: loading data
print('\nLoading data samples')
Xdata_train, Ydata_train, Xdata_test, Ydata_test, \
    Lat_train, Lon_train, Lat_test, Lon_test = load_data(data_file)

n_samples_testing,ny,nx,nchans = Xdata_test.shape

# Step 2: loading ANN model
print('Loading ANN model')
model = load_model(modelfile, \
    custom_objects={"my_r_square_metric": my_r_square_metric})
print('\n')
print(model.summary())
print('\n')

# Add flatten layer at the end to be able to use visualization
if IS_UNET:
    # print('UNET not yet implemented.  To come soon.\n')
    input = model.input
    output = model.output
    output = Flatten()(output)
    model = models.Model(inputs=input, outputs=output)

else:
    model.add(Flatten())

print(model.summary())


##############################################
######## PREDICTIONS ##############
# Step 3: generate predictions
print('\nGenerate predictions')
if ALSO_PREDICT_TRAINING_DATA:
   Zdata_train = model.predict(Xdata_train)
   Zdata_train = Zdata_train.reshape(Zdata_train.shape[0], nx, ny)  # reverse the flattening

Zdata_test = model.predict(Xdata_test)
Zdata_test = Zdata_test.reshape(Zdata_test.shape[0], nx, ny)  # reverse the flattening

# Step 4: RESTORE ORIGINAL SCALING - FOR OUTPUT CHANNEL ONLY ###
print('\nRestore original scaling')
if ALSO_PREDICT_TRAINING_DATA:
    Zdata_train = np.array(Zdata_train,dtype=np.float64)
    Zdata_train *= ymax
    Ydata_train *= ymax
    print('Zdata_train min,mean,max=', np.min(Zdata_train), np.mean(Zdata_train), np.max(Zdata_train))
    print('Ydata_train min,mean,max=', np.min(Ydata_train), np.mean(Ydata_train), np.max(Ydata_train))

Zdata_test = np.array(Zdata_test,dtype=np.float64)
Zdata_test *= ymax
Ydata_test *= ymax
print('Zdata_test min,mean,max=',np.min(Zdata_test),np.mean(Zdata_test),np.max(Zdata_test))
print('Ydata_test min,mean,max=',np.min(Ydata_test),np.mean(Ydata_test),np.max(Ydata_test))


# Done with preparation steps for data and predictions.

###############################################
############## VISUALIZATION# #################
###############################################

############### Visualization Task #1 ###############
# Generate convergence plot
#if WANT_CONVERGENCE_PLOT:
#    print('Convergence plots')
#    plot_convergence(historyfile,convergence_plot_file)

###############################################

# Only set ONE of the following to true.
# Otherwise the "last" method with True will be used to generate map.

# Not very elegant...  improve later.
WANT_Occlusion_HEATMAP = False  # This method did not seem useful.
WANT_GradCAM_HEATMAP = False  # This method did not seem useful either.
WANT_SmoothGrad_HEATMAP = True  # This one seems to give a good estimate of effective receptive field.


if True:

    ###############################################
    ### Get the sample you want to analyze
    i_testsample = 0
    my_sample = Xdata_test[i_testsample,:,:,:]
    n_samples_to_analyze = 1
    my_sample = my_sample.reshape(nx,ny,nchans)

    ###############################################
    # Choose which pixel in output image we want to analyze.
    my_row=100 #25
    my_col=100 #120
    my_index = my_row * ny + my_col   # Should this be nx or ny?  Doesn't matter here, because nx=ny.

    ### Check we used to do for LRP - should not be needed here, but printing value anyway:
    pixel_value = Zdata_test[i_testsample, my_row, my_col]  # pixel value of estimate
    print('Pixel(' + repr(my_row) + ',' + repr(my_col) + ') = ' + repr(pixel_value))
    # if pixel_value <= 0:
    #    print( '\n --- Warning - estimate for LRP might not be correct! --- \n --- Reason: estimated output at pixel={}. ---'.format(pixel_value))

    my_data = ( [my_sample], None )

    # CV colormaps to choose from:
    # See https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html#gga9a805d8262bcbe273f16be9ea2055a65afdb81862da35ea4912a75f0e8f274aeb
    #   for list of CV colormaps
    # COLORMAP_HOT  # white to red
    # COLORMAP_BONE # white to gray
    # COLORMAP_PINK # white to reddish


    if WANT_Occlusion_HEATMAP:  # Occlusion method - results were disappointing.

        ########### Method: Occlusion ################
        explainer = OcclusionSensitivity() # define which type of heatmap we want
        occlusion_patch_width = 20
        heatmap = explainer.explain( my_data, model, my_index, occlusion_patch_width, colormap=cv2.COLORMAP_PINK )

        my_title_text = 'Occlusion - patch = ' + repr(occlusion_patch_width) + ' pixels'
        my_file_text = 'occlusion' + '_' + repr(occlusion_patch_width)  #+ repr{my_row} + '{}_P{}_{}'.format('%i %i %i')

    if WANT_GradCAM_HEATMAP:  # GradCAM - results made no sense either
        ########### Method: Grad CAM ################
        explainer = GradCAM()
        my_layer_name = 'conv2d_2'
        heatmap = explainer.explain(my_data, model, my_layer_name, my_index, colormap=cv2.COLORMAP_HOT )

        my_title_text = 'GradCam - Layer:' + my_layer_name
        my_file_text = 'GradCam_'  + my_layer_name


    if WANT_SmoothGrad_HEATMAP:
        ########### Method: SmoothGrad ################
        print( 'Note: if num_samples is chosen large, then this can take several minutes\n')
        explainer = SmoothGrad()
        my_layer_name = 'conv2d_2'
        num_samples = 20 #200
        noise = 1.0  #1.0
        heatmap = explainer.explain(my_data, model, my_index, num_samples, noise)   #, colormap=cv2.COLORMAP_HOT)

        my_title_text = 'SmoothGrad - samples ' + repr(num_samples) + ' noise ' + repr(noise)
        my_file_text = 'SmoothGrad_s' + repr(num_samples) + '_n' + repr(noise)


    ########## VISUALIZE RESULTS ###################

    # Show heatmap by itself.
    f, axes = plt.subplots(1, 1, figsize=(8, 8))
    axes.imshow(heatmap)
    axes.set_title('Heatmap for row={} col={}.\n{}'.format(my_row, my_col, my_title_text) )

    # Save heatmap to file.
    my_plot_filename = heat_map_file_name_start(IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs) + '_' + my_file_text  + '.png'
    plt.savefig(my_plot_filename)
    plt.close()
    print('Saved file to ' + my_plot_filename + '\n')

    #################




