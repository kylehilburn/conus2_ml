defcon = {}

defcon['NN_string'] = 'SEQ'  #'SEQ' or 'UNET'
defcon['activ'] = 'relu'
defcon['activ_last'] = 'linear'
defcon['batch_step_size'] = 100  # 100 steps per epoch
defcon['batchnorm'] = False
defcon['convfilter'] = (3,3)
defcon['convfilter_last_layer'] = (1,1)
defcon['data_suffix'] = ''
defcon['double_filters'] = True  #double filters for each layer
defcon['dropout'] = False
defcon['dropout_rate'] = 0.1
defcon['kernel_init'] = 'glorot_uniform'  # default in TF
defcon['loss'] = 'mean_squared_error'
defcon['machine'] = 'notHera'
defcon['n_conv_layers_per_decoder_layer'] = 1  #1=CP blocks, 2=CCP,...; for up-blocks
defcon['n_conv_layers_per_encoder_layer'] = 1  #1=CP blocks, 2=CCP,...
defcon['n_encoder_decoder_layers'] = 3 # choose integer - 0,1,2,3,...
defcon['n_filters_for_first_layer'] = 8
defcon['n_filters_last_layer'] = 1
defcon['nepochs'] = 50
defcon['poolfilter'] = (2,2)
defcon['upfilter'] = (2,2)
