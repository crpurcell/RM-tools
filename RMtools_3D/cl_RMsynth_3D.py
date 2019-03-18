#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     cl_RMsynth_3D.py                                                  #
#                                                                             #
# PURPOSE:  Run RM-synthesis on a Stokes Q & U cubes.                         #
#                                                                             #
# MODIFIED: 7-March-2019 by J. West                                           #
#                                                                             #
#=============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2016 Cormac R. Purcell                                        #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the "Software"),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
#=============================================================================#

import sys
import os
import time
import argparse
import math as m
import numpy as np
import astropy.io.fits as pf
import pdb

from RMutils.util_RM import do_rmsynth_planes
from RMutils.util_RM import get_rmsf_planes
from RMutils.util_misc import interp_images

C = 2.997924538e8 # Speed of light [m/s]

#-----------------------------------------------------------------------------#
def run_rmsynth(dataQ, dataU, freqArr_Hz, headtemplate, dataI=None, rmsArr_Jy=None,
                phiMax_radm2=None, dPhi_radm2=None, nSamples=10.0,
                weightType="uniform", prefixOut="", outDir="",
                fitRMSF=False, nBits=32, write_seperate_FDF=False, verbose=True):

    """Read the Q & U data and run RM-synthesis."""
    # Sanity check on header dimensions

    if not str(dataQ.shape) == str(dataU.shape):
        print("Err: unequal dimensions: Q = "+str(dataQ.shape)+", U = "+str(dataU.shape)+".")
        sys.exit()

    # Check dimensions of Stokes I cube, if present
    if not dataI is None:
        if not str(dataI.shape) == str(dataQ.shape):
            print("Err: unequal dimensions: Q = "+str(dataQ.shape)+", I = "+str(dataI.shape)+".")
            sys.exit()
    
    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)

    lambdaSqArr_m2 = np.power(C/freqArr_Hz, 2.0)

    dFreq_Hz = np.nanmin(np.abs(np.diff(freqArr_Hz)))
    lambdaSqRange_m2 = ( np.nanmax(lambdaSqArr_m2) -
                         np.nanmin(lambdaSqArr_m2) )        
    dLambdaSqMin_m2 = np.nanmin(np.abs(np.diff(lambdaSqArr_m2)))
    dLambdaSqMax_m2 = np.nanmax(np.abs(np.diff(lambdaSqArr_m2)))
    
    # Set the Faraday depth range
    fwhmRMSF_radm2 = 2.0 * m.sqrt(3.0) / lambdaSqRange_m2
    if dPhi_radm2 is None:
        dPhi_radm2 = fwhmRMSF_radm2 / nSamples
    if phiMax_radm2 is None:
        phiMax_radm2 = m.sqrt(3.0) / dLambdaSqMax_m2
        phiMax_radm2 = max(phiMax_radm2, 600.0)    # Force the minimum phiMax

    # Faraday depth sampling. Zero always centred on middle channel
    nChanRM = round(abs((phiMax_radm2 - 0.0) / dPhi_radm2)) * 2.0 + 1.0
    startPhi_radm2 = - (nChanRM-1.0) * dPhi_radm2 / 2.0
    stopPhi_radm2 = + (nChanRM-1.0) * dPhi_radm2 / 2.0
    phiArr_radm2 = np.linspace(startPhi_radm2, stopPhi_radm2, nChanRM)
    phiArr_radm2 = phiArr_radm2.astype(dtFloat)
    if(verbose): print("PhiArr = %.2f to %.2f by %.2f (%d chans)." % (phiArr_radm2[0],
                                                        phiArr_radm2[-1],
                                                        float(dPhi_radm2),
                                                        nChanRM))
    
        
    # Calculate the weighting as 1/sigma^2 or all 1s (uniform)
    if weightType=="variance" and rmsArr_Jy is not None:
        weightArr = 1.0 / np.power(rmsArr_Jy, 2.0)
    else:
        weightType = "uniform"
        weightArr = np.ones(freqArr_Hz.shape, dtype=dtFloat)    
    if(verbose): print("Weight type is '%s'." % weightType)

    startTime = time.time()

    # Read the Stokes I model and divide into the Q & U data
    if dataI:    
        with np.errstate(divide='ignore', invalid='ignore'):
            qArr = np.true_divide(dataQ, dataI)
            uArr = np.true_divide(dataU, dataI)
    else:
        qArr = dataQ
        uArr = dataU
        
    # Perform RM-synthesis on the cube
    FDFcube, lam0Sq_m2 = do_rmsynth_planes(dataQ           = qArr,
                                           dataU           = uArr,
                                           lambdaSqArr_m2  = lambdaSqArr_m2,
                                           phiArr_radm2    = phiArr_radm2,
                                           weightArr       = weightArr,
                                           nBits           = 32,
                                           verbose         = True)
    # Calculate the Rotation Measure Spread Function cube
    RMSFcube, phi2Arr_radm2, fwhmRMSFCube, fitStatArr = \
        get_rmsf_planes(lambdaSqArr_m2   = lambdaSqArr_m2,
                        phiArr_radm2     = phiArr_radm2,
                        weightArr        = weightArr,
                        mskArr           = ~np.isfinite(dataQ),
                        lam0Sq_m2        = lam0Sq_m2,
                        double           = True,
                        fitRMSF          = fitRMSF,
                        fitRMSFreal      = False,
                        nBits            = 32,
                        verbose          = True)
    endTime = time.time()
    cputime = (endTime - startTime)
    if(verbose): print("> RM-synthesis completed in %.2f seconds." % cputime)
    if(verbose): print("Saving the dirty FDF, RMSF and ancillary FITS files.")

    # Determine the Stokes I value at lam0Sq_m2 from the Stokes I model
    # Note: the Stokes I model MUST be continuous throughout the cube,
    # i.e., no NaNs as the amplitude at freq0_Hz is interpolated from the
    # nearest two planes.
    freq0_Hz = C / m.sqrt(lam0Sq_m2)
    if dataI:
        idx = np.abs(freqArr_Hz - freq0_Hz).argmin()
        if freqArr_Hz[idx]<freq0_Hz:
            Ifreq0Arr = interp_images(dataI[idx, :, :], dataI[idx+1, :, :], f=0.5)
        elif freqArr_Hz[idx]>freq0_Hz:
            Ifreq0Arr = interp_images(dataI[idx-1, :, :], dataI[idx, :, :], f=0.5)
        else:
            Ifreq0Arr = dataI[idx, :, :]

        # Multiply the dirty FDF by Ifreq0 to recover the PI in Jy
        FDFcube *= Ifreq0Arr
    
    # Make a copy of the Q header and alter Z-axis as Faraday depth
    header = headtemplate
    header["CTYPE3"] = "FARADAY DEPTH"
    header["CDELT3"] = np.diff(phiArr_radm2)[0]
    header["CRPIX3"] = 1.0
    header["CRVAL3"] = phiArr_radm2[0]
    if "DATAMAX" in header:
        del header["DATAMAX"]
    if "DATAMIN" in header:
        del header["DATAMIN"]

    if outDir=='':  #To prevent code breaking if file is in current directory
        outDir='.'


    
    hdu0 = pf.PrimaryHDU(FDFcube.real.astype(dtFloat), header)
    hdu1 = pf.ImageHDU(FDFcube.imag.astype(dtFloat), header)
    hdu2 = pf.ImageHDU(np.abs(FDFcube).astype(dtFloat), header)
    if(write_seperate_FDF):
        fitsFileOut = outDir + "/" + prefixOut + "FDF_real_dirty.fits"
        if(verbose): print("> %s" % fitsFileOut)
        hdu0.writeto(fitsFileOut, output_verify="fix", overwrite=True)

        fitsFileOut = outDir + "/" + prefixOut + "FDF_im_dirty.fits"
        if(verbose): print("> %s" % fitsFileOut)
        hdu1.writeto(fitsFileOut, output_verify="fix", overwrite=True)

        fitsFileOut = outDir + "/" + prefixOut + "FDF_tot_dirty.fits"
        if(verbose): print("> %s" % fitsFileOut)
        hdu2.writeto(fitsFileOut, output_verify="fix", overwrite=True)

    else:
        # Save the dirty FDF
        fitsFileOut = outDir + "/" + prefixOut + "FDF_dirty.fits"
        if(verbose): print("> %s" % fitsFileOut)
        hduLst = pf.HDUList([hdu0, hdu1, hdu2])
        hduLst.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        hduLst.close()
    
    # Save a maximum polarised intensity map
    fitsFileOut = outDir + "/" + prefixOut + "FDF_maxPI.fits"
    if(verbose): print("> %s" % fitsFileOut)
    pf.writeto(fitsFileOut, np.max(np.abs(FDFcube), 0).astype(dtFloat), header,
               overwrite=True, output_verify="fix")
    
    # Save a peak RM map
    fitsFileOut = outDir + "/" + prefixOut + "FDF_peakRM.fits"
    header["BUNIT"] = "rad/m^2"
    peakFDFmap = np.argmax(np.abs(FDFcube), 0).astype(dtFloat)
    peakFDFmap = header["CRVAL3"] + (peakFDFmap + 1
                                     - header["CRPIX3"]) * header["CDELT3"]
    if(verbose): print("> %s" % fitsFileOut)
    pf.writeto(fitsFileOut, peakFDFmap, header, overwrite=True,
               output_verify="fix")
    
    # Save an RM moment-1 map
    fitsFileOut = outDir + "/" + prefixOut + "FDF_mom1.fits"
    header["BUNIT"] = "rad/m^2"
    mom1FDFmap = (np.nansum(np.abs(FDFcube).transpose(1,2,0) * phiArr_radm2, 2)
                  /np.nansum(np.abs(FDFcube).transpose(1,2,0), 2))
    mom1FDFmap = mom1FDFmap.astype(dtFloat)
    if(verbose): print("> %s" % fitsFileOut)
    pf.writeto(fitsFileOut, mom1FDFmap, header, overwrite=True,
               output_verify="fix")

    # Save the RMSF
    header["CRVAL3"] = phi2Arr_radm2[0]
    fitsFileOut = outDir + "/" + prefixOut + "RMSF.fits"
    hdu0 = pf.PrimaryHDU(RMSFcube.real.astype(dtFloat), header)
    hdu1 = pf.ImageHDU(RMSFcube.imag.astype(dtFloat), header)
    hdu2 = pf.ImageHDU(np.abs(RMSFcube).astype(dtFloat), header)
    header["DATAMAX"] = np.max(fwhmRMSFCube) + 1
    header["DATAMIN"] = np.max(fwhmRMSFCube) - 1
    hdu3 = pf.ImageHDU(fwhmRMSFCube.astype(dtFloat), header)
    hduLst = pf.HDUList([hdu0, hdu1, hdu2, hdu3])
    if(verbose): print("> %s" % fitsFileOut)
    hduLst.writeto(fitsFileOut, output_verify="fix", overwrite=True)
    hduLst.close()

def readFitsCube(file, verbose):

    if not os.path.exists(file):
        print("Err: File not found")
    
    if(verbose): print("Reading " + file + " ...", end=' ')    
    data = pf.getdata(file)
    head = pf.getheader(file)
    if(verbose): print("done.")
    
    if head['CTYPE3']=='FREQ': 
        freqAx=3
        data=data[:,:,:]
        # Feeback
        if(verbose): print("The first 3 dimensions of the cubes are [X=%d, Y=%d, Z=%d]." % \
          (headQ["NAXIS1"], headQ["NAXIS2"], headQ["NAXIS3"]))

    elif head["NAXIS"]==4:
        # Feeback
        if(verbose): print("The first 4 dimensions of the cubes are [X=%d, Y=%d, Z=%d, F=%d]." % \
          (head["NAXIS1"], head["NAXIS2"], head["NAXIS3"], head["NAXIS4"]))
        if(head['CTYPE4']=='FREQ'): 
            freqAx=4
            data=data[:,0,:,:]
        else: print("Err: No frequency axis found")

    return head, data
    
def readFreqFile(file, verbose):
    # Read the frequency vector and wavelength sampling
    freqArr_Hz = np.loadtxt(file, dtype=float)
    return freqArr_Hz
