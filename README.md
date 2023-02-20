# conus2_ml

GREMLIN conus2_ml repository README
Kyle Hilburn, 20-Feb-2023

Code was written by Kyle Hilburn and Imme Ebert-Uphoff.

The GREMLIN CONUS2 dataset consists of 92 “cases” where each case is a six-hour period of severe weather (producing SPC storm reports) sampled every 15-minutes. The cases cover a 92-day period from April to July 2019. The first 74 cases were used for training and the last 18 cases were used for testing. Each case was stored in its own NetCDF file. The NetCDF files were produced by resampling GOES-16 ABI, GOES-16 GLM, and MRMS Composite Reflectivity onto the HRRR CONUS mass grid (3 km) and extracting 256 x 256-pixel image patches to maximize the storm reports.

MAIN_PREPARE_SAVE_DATA.py reads the cases, scales the data, and outputs an NPZ file. This was run on NOAA Hera using run_PREPARE.sh.

MAIN_TRAIN_and_SAVE_MODEL.py reads the NPZ data file and a plain text configuration file and then uses TensorFlow to train a model. The output is a trained model (HDF5) and the training history (pickled binary). TensorFlow Version 1 was used. This was run on NOAA Hera using run_TRAIN.sh

The “GREMLIN Version-1 CNN” is:
	model_K12_WTD_ALL_3x3_T_SEQ_blocks_3_epochs_100.h5
and had the following configuration:
	NN_string='SEQ'
	nepochs=100
	batch_size=18
	n_filters_for_first_layer=32
	double_filters=False
	loss='my_mean_squared_error_weighted_genexp'
	loss_weight=(1.0, 5.0, 4.0)
	my_file_prefix='K12_WTD_ALL_3x3_T'
	data_suffix = 'rescaleC09'

MAIN_POST_PROCESSING.py was used after training to generate predictions, print statistics, and plot convergence plots.

MAIN_VISUALIZATION.py was used to create backwards optimization maps, feature maps, feature activations, and the effective receptive field.

MAIN_VIS_LRP.py was used to create Layerwise Relevance Propagation (LRP) heat maps.

MAIN_VIS_tf_explain.py uses tf-explain was used for Occlusion, GradCam, and SmoothGrad XAI.
