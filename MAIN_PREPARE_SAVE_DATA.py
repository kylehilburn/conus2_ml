import numpy as np
import sys
from datetime import datetime

# Read data and pre-process (normalization, etc.)
from prepare_data import prepare_data
from make_custom_file_names import data_file_name
from read_configuration import read_configuration
from default_configuration import defcon

print('start MAIN_PREPARE_SAVE_DATA=',datetime.now())

# required command line argument: my_file_prefix
try:
    my_file_prefix = sys.argv[1]
except IndexError:
    sys.exit('Error: you must supply my_file_prefix as command line argument')
print('my_file_prefix =',my_file_prefix)

config = read_configuration()

try:
    machine = config['machine']
except KeyError:
    try:
        machine = config[my_file_prefix]['machine']
    except KeyError:
        machine = defcon['machine']
print('machine =',machine)

spath = '..'  #Imme
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

data_file = data_file_name( spath, suffix=data_suffix ) # get data file name

filename_format = spath+'/SAMPLES/case{0:02n}.nc'
ncases = 92
ncases_test = 18
cases_train = [i+1 for i in range(ncases-ncases_test)]
cases_test = [i+ncases-ncases_test+1 for i in range(ncases_test)]
channels = ['C07','C09','C13','GROUP']
qctimes_train = [] #all good
qctimes_test = [] #all good

if data_suffix == 'LTG1':
    xmin = {1:0.00, 6:0.00, 7:200, 9:200, 13:200, 'GROUP':0.0}
    xmax = {1:1.00, 6:1.00, 7:300, 9:300, 13:300, 'GROUP':50.0}
    train = prepare_data(filename_format, cases_train, channels, qctimes_train, xmin=xmin,xmax=xmax)
    test = prepare_data(filename_format, cases_test, channels, qctimes_test, xmin=xmin,xmax=xmax)
else:
    train = prepare_data(filename_format, cases_train, channels, qctimes_train)
    test = prepare_data(filename_format, cases_test, channels, qctimes_test)

if data_suffix == 'C13':
    print('zero out C07, C09, GLM')
    train['Xdata'][:,:,:,0] = 0.  #zero out C07
    train['Xdata'][:,:,:,1] = 0.  #zero out C09
    train['Xdata'][:,:,:,3] = 0.  #zero out GLM
    test['Xdata'][:,:,:,0] = 0.  #zero out C07
    test['Xdata'][:,:,:,1] = 0.  #zero out C09
    test['Xdata'][:,:,:,3] = 0.  #zero out GLM

if data_suffix == 'C13GLM':
    print('zero out C07, C09')
    train['Xdata'][:,:,:,0] = 0.  #zero out C07
    train['Xdata'][:,:,:,1] = 0.  #zero out C09
    test['Xdata'][:,:,:,0] = 0.  #zero out C07
    test['Xdata'][:,:,:,1] = 0.  #zero out C09

if data_suffix == 'C13C09':
    print('zero out C07, GLM')
    train['Xdata'][:,:,:,0] = 0.  #zero out C07
    train['Xdata'][:,:,:,3] = 0.  #zero out GLM
    test['Xdata'][:,:,:,0] = 0.  #zero out C07
    test['Xdata'][:,:,:,3] = 0.  #zero out GLM

if data_suffix == 'C13C09GLM':
    print('zero out C07')
    train['Xdata'][:,:,:,0] = 0.  #zero out C07
    test['Xdata'][:,:,:,0] = 0.  #zero out C07

print('Saving data to file:' + data_file)
np.savez( data_file, Xdata_train=train['Xdata'], Ydata_train=train['Ydata'],
   Xdata_test=test['Xdata'], Ydata_test=test['Ydata'],
   Lat_train=train['Lat'], Lon_train=train['Lon'],
   Lat_test=test['Lat'], Lon_test=test['Lon'] )

print('end MAIN_PREPARE_SAVE_DATA=',datetime.now())
