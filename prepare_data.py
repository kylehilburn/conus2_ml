from netCDF4 import Dataset, num2date
import numpy as np
import sys
#from read_mrms_coverage_mask import read_mask
from copy import deepcopy

tmatch_sec_default = 5.*60.

nx_default = 256
ny_default = 256

#covmask = read_mask()
covthrpct_default = 50.

yvar_default = 'MRMS_REFC'
ymin_default = 0.0
ymax_default = 60.0

xmin_default = {1:0.00, 6:0.00, 7:200, 9:200, 13:200, 'GROUP':0.1}
xmax_default = {1:1.00, 6:1.00, 7:300, 9:300, 13:300, 'GROUP':50.0}

stds_default = {'y':1., 2:0.01, 6:0.01, 7:1., 9:1., 13:1., 'GROUP':0.1}

solar_channels = [1,2,3,4,5,6]

def prepare_data(filename_format, caselist, channels, qctimes, \
    augfac = 0, \
    verbose = False, \
    fillvalue = 0, \
    nx = nx_default, \
    ny = ny_default, \
    szanorm = False, \
    covthrpct = covthrpct_default, \
    yvar = yvar_default, \
    ymin = ymin_default, \
    ymax = ymax_default, \
    xmin = xmin_default, \
    xmax = xmax_default, \
    stds = stds_default, \
    tmatch_sec = tmatch_sec_default, \
    ):

#   required arguments:
#     filename_format = format string to generate case file names
#     caselist = list of case numbers
#     channels = list of channels, e.g., ['C02','C13','GROUP']
#     qctimes = list of datetimes to not use
#
#   optional arguments:
#     augfac = how many augmented samples per real samples
#              (default=0=no augmentation)
#     verbose = default=False, if True then print more information
#     fillvalue = value to fill bad/missing data in Xdata,Ydata
#     nx = x-dimension value (integer)
#     ny = y-dimension value (integer)
#     szanorm = default=False, if true then normalize C01-C06 by 
#               solar zenith angle
#
#   "expert" optional arguments (should not have to change):
#     covthrpct = minimum radar coverage, percent (float)
#     yvar = Ydata variable name (string)
#     ymin = Ydata normalization minimum (float)
#     ymax = Ydata normalization maximum (float)
#     xmin = Xdata normalization minimum (dictionary)
#     xmax = Xdata normalization maximum (dictionary)
#     stds = augmentation standard deviations for x and y variables
#            (dictionary)
#     tmatch_sec = time radius for qctimes in seconds (float)
#
#   usage notes:
#     bad/missing radar data can either be -999, NaN, or masked
#     bad/missing satellite data can either be -1.E30, NaN, or masked
#     Channels for ABI should have names like 'C02'
#     Channels for GLM should have names like 'GROUP' or 'FLASH'
#     ABI channels 1-6 are mapped from 0 to 1
#     ABI channels 7-16 are mapped from 1 to 0
#
#   returns:
#     dictionary with following keys:
#       Xdata = predictors, array(nbatches,ny,nx,nchans)
#       Ydata = predictand, array(nbatches,ny,nx)
#       nsamples = number of real samples
#       nbatches = total number of samples, including augmented
#       Lat = latitude, array(nsamples,ny,nx)
#       Lon = longitude, array(nsamples,ny,nx)
#       Shape = shape for input layer of NN
#       Dates = list of datetime objects for each sample
#
#  development notes:
#     solar reflective bands SZA normalization is not fully implemented
#       (see STOP)
#     radar coverage masking has been commented out - 
#       need to treat moving/stationary ROIs


# step 1: unpack channel information

    nchans = len(channels)

    shape = (ny,nx,nchans)

    datafile = filename_format.format(caselist[0])

    ds = Dataset(datafile)

    cindices = []  #channel index in datafile
    cvalues = []  #channel number (C07=7) or name (GLM_GROUPS)
    ctypes = []
    for achan in channels:
        if achan.startswith('C'):
            ihit = np.argmin(np.abs((ds.variables['channels'][:]-\
                int(achan.replace('C','')))))
            cindices.append(ihit)
            cvalues.append(ds.variables['channels'][:][ihit])
            ctypes.append('ABI')
        else:
            cindices.append(None)
            cvalues.append(achan)
            ctypes.append('GLM')
    if verbose:
        print('cindices=',cindices)
        print('cvalues=',cvalues)
        print('ctypes=',ctypes)
        #print('dataset channels=',ds.variables['channels'][:])

    ds.close()

# step 2: determine number of batches 

    nbatches = 0

    for acase in caselist:

        datafile = filename_format.format(acase)

        ds = Dataset(datafile)

        times = ds.variables['time']
        dates = num2date(times[:],units=times.units,calendar=times.calendar)

        for idate,adate in enumerate(dates):

            isbad = False

            yflag = ds.variables[yvar+'_FLAG'][idate]
            if yflag != 0:
                if verbose:
                    print('missing mrms at '+\
                        adate.strftime('%Y-%m-%d_%H:%MZ'))
                isbad = True

            for ichan,achan,chant,kchan in \
                zip(cindices,cvalues,ctypes,range(nchans)):
                if chant == 'ABI':
                    xflag = ds.variables['GOES_ABI_FLAG'][idate,ichan]
                else:
                    xflag = ds.variables['GOES_GLM_'+achan+'_FLAG'][idate]
                if xflag != 0:
                    if verbose:
                        print('missing goes for '+str(achan)+' at '+\
                            adate.strftime('%Y-%m-%d_%H:%MZ'))
                    isbad = True

            for aqctime in qctimes:
                if np.abs((adate-aqctime).total_seconds()) < tmatch_sec:
                    isbad = True

            if not isbad:
                nbatches += 1

        ds.close()

    nsamples = deepcopy(nbatches)  #number of real samples
    nbatches *= (augfac+1)
    if verbose:
        print('nbatches=',nbatches)
        print('nsamples=',nsamples)

    Xdata = np.zeros((nbatches,ny,nx,nchans))
    Ydata = np.zeros((nbatches,ny,nx))
    datelist = []

    #only store for real samples:
    lat = np.zeros((nsamples,ny,nx))
    lon = np.zeros((nsamples,ny,nx))
    badmask = np.zeros((nsamples,ny,nx),dtype=np.int32)  #0=good, 1=bad

# step 3: fill arrays with real samples

    ibatch = 0

    for acase in caselist:

        datafile = filename_format.format(acase)

        ds = Dataset(datafile)

        times = ds.variables['time']
        dates = num2date(times[:],units=times.units,calendar=times.calendar)

        for idate,adate in enumerate(dates):

            isbad = False

            yflag = ds.variables[yvar+'_FLAG'][idate]
            if yflag != 0:
                if verbose:
                    print('missing mrms at '+\
                        dates[idate].strftime('%Y-%m-%d_%H:%MZ'))
                isbad = True

            for ichan,achan,chant,kchan in \
                zip(cindices,cvalues,ctypes,range(nchans)):
                if chant == 'ABI':
                    xflag = ds.variables['GOES_ABI_FLAG'][idate,ichan]
                else:
                    xflag = ds.variables['GOES_GLM_'+achan+'_FLAG'][idate]
                if xflag != 0:
                    if verbose:
                        print('missing goes for '+str(achan)+' at '+\
                            adate.strftime('%Y-%m-%d_%H:%MZ'))
                    isbad = True

            for aqctime in qctimes:
                if np.abs((adate-aqctime).total_seconds()) < tmatch_sec:
                    isbad = True

            if isbad: continue

# need this when box is moving:
#            lat[ibatch,:,:] = ds.variables['lat'][idate,:,:]
#            lon[ibatch,:,:] = ds.variables['lon'][idate,:,:]
# stationary box:
            lat[ibatch,:,:] = ds.variables['lat'][:,:]
            lon[ibatch,:,:] = ds.variables['lon'][:,:]

            Ydata[ibatch,:,:] = ds.variables[yvar][idate,:,:]

            datelist.append(adate)

            for ichan,achan,chant,kchan in \
                zip(cindices,cvalues,ctypes,range(nchans)):
                if chant == 'ABI':
                    Xdata[ibatch,:,:,kchan] = \
                        ds.variables['GOES_ABI'][idate,ichan,:,:]
                else:
                    Xdata[ibatch,:,:,kchan] = \
                        ds.variables['GOES_GLM_'+achan][idate,:,:]

# I forgot to include these variables in latest run,
# dimension of which will also depend on moving or stationary box
#            ilon1 = ds.variables['illlon'][idate]
#            ilon2 = ds.variables['iurlon'][idate]
#            ilat1 = ds.variables['illlat'][idate]
#            ilat2 = ds.variables['iurlat'][idate]
#
#            badmask[ibatch,:,:][\
#                (covmask[ilat1:ilat2,ilon1:ilon2] < covthrpct)] = 1
            badmask[ibatch,:,:][(Ydata[ibatch,:,:] == -999)] = 1
            badmask[ibatch,:,:][(np.ma.getmask(Ydata[ibatch,:,:]))] = 1
            badmask[ibatch,:,:][~np.isfinite(Ydata[ibatch,:,:])] = 1
            for ichan in range(nchans):
                badmask[ibatch,:,:][\
                    (Xdata[ibatch,:,:,ichan] == -1.E30)] = 1
                badmask[ibatch,:,:][\
                    (np.ma.getmask(Xdata[ibatch,:,:,ichan]))] = 1
                badmask[ibatch,:,:][\
                    ~np.isfinite(Xdata[ibatch,:,:,ichan])] = 1
            if verbose:
                print('ibatch,adate,nbad,ngood=',ibatch,\
                    adate.strftime('%Y-%m-%d %H:%M:%SZ'),\
                    np.sum(badmask[ibatch,:,:]==1),\
                    np.sum(badmask[ibatch,:,:]==0))

            if szanorm:
                sys.exit('STOP: not implemented - '\
                    'need to have sza threshold (e.g., bad when > 88 deg)'\
                    ' and update badmask')
#            for ichan,achan,chant,kchan in \
#                zip(cindices,cvalues,ctypes,range(nchans)):
#                if chant == 'ABI':
#                    if szanorm and (achan in solar_channels):
#                        sza = ds.variables['Solar_Zenith'][idate,:,:]
#                        Xdata[ibatch,:,:,kchan] /= np.cos(np.radians(sza))

            ibatch += 1

# step 4: augmentation with random noise

    for iaug in range(augfac+1):
        if iaug == 0: continue  #don't augment real samples
        for isam in range(nsamples):
            ibatch = iaug*nsamples + isam
            Ydata[ibatch,:,:] = Ydata[isam,:,:] + \
                np.random.normal(loc=0.0,scale=stds['y'],size=(ny,nx))
            for ichan,achan in enumerate(cvalues):
                Xdata[ibatch,:,:,ichan] = Xdata[isam,:,:,ichan] + \
                    np.random.normal(loc=0.0,scale=stds[achan],size=(ny,nx))

# step 5: normalize, clip, mask

    for iaug in range(augfac+1):
        for isam in range(nsamples):
            ibatch = iaug*nsamples + isam

            Ydata[ibatch,:,:] = (Ydata[ibatch,:,:]-ymin)/(ymax-ymin)
            Ydata[ibatch,:,:][Ydata[ibatch,:,:]<0] = 0.
            Ydata[ibatch,:,:][Ydata[ibatch,:,:]>1] = 1.
            Ydata[ibatch,:,:][badmask[isam,:,:]==1] = fillvalue

            for ichan,achan,chant in zip(range(nchans),cvalues,ctypes):
                if chant == 'GLM':
                    Xdata[ibatch,:,:,ichan] = \
                        (Xdata[ibatch,:,:,ichan]-xmin[achan])/\
                        (xmax[achan]-xmin[achan])
                    Xdata[ibatch,:,:,ichan][\
                        Xdata[ibatch,:,:,ichan]<0] = 0.
                    Xdata[ibatch,:,:,ichan][\
                        Xdata[ibatch,:,:,ichan]>1] = 1.
                    Xdata[ibatch,:,:,ichan][\
                        badmask[isam,:,:]==1] = fillvalue
                else:
                    if achan < 7:
                        Xdata[ibatch,:,:,ichan] = \
                            (Xdata[ibatch,:,:,ichan]-xmin[achan])/\
                            (xmax[achan]-xmin[achan])
                        Xdata[ibatch,:,:,ichan][\
                            Xdata[ibatch,:,:,ichan]<0] = 0.
                        Xdata[ibatch,:,:,ichan][\
                            Xdata[ibatch,:,:,ichan]>1] = 1.
                        Xdata[ibatch,:,:,ichan][\
                            badmask[isam,:,:]==1] = fillvalue
                    else:
                        Xdata[ibatch,:,:,ichan] = \
                            (xmax[achan]-Xdata[ibatch,:,:,ichan])/\
                            (xmax[achan]-xmin[achan])
                        Xdata[ibatch,:,:,ichan][\
                            Xdata[ibatch,:,:,ichan]<0] = 0.
                        Xdata[ibatch,:,:,ichan][\
                            Xdata[ibatch,:,:,ichan]>1] = 1.
                        Xdata[ibatch,:,:,ichan][\
                            badmask[isam,:,:]==1] = fillvalue

    ds.close()

    outdict = {}
    outdict['Xdata'] = Xdata
    outdict['Ydata'] = Ydata
    outdict['nsamples'] = nsamples
    outdict['nbatches'] = nbatches
    outdict['Lat'] = lat
    outdict['Lon'] = lon
    outdict['Shape'] = shape
    outdict['Dates'] = datelist

    return outdict
