#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     cl_RMclean_1D.py                                                  #
#                                                                             #
# PURPOSE:  Command line functions for RM-clean                               #
#           on a dirty Faraday dispersion function.                           #
# CREATED:  16-Nov-2018 by J. West                                            #
# MODIFIED: 16-Nov-2018 by J. West                                            #
#                                                                             #
#=============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2015 Cormac R. Purcell                                        #
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
import json
import math as m
import numpy as np
import pdb

from RMutils.util_RM import do_rmclean
from RMutils.util_RM import do_rmclean_hogbom
from RMutils.util_RM import measure_FDF_parms
from RMutils.util_RM import measure_fdf_complexity

C = 2.997924538e8 # Speed of light [m/s]

#-----------------------------------------------------------------------------#
def run_rmclean(mDictS, aDict, cutoff,
                maxIter=1000, gain=0.1, prefixOut="", outDir="", nBits=32,
                showPlots=False, doAnimate=False, verbose=False):
    """
    Run RM-CLEAN on a complex FDF spectrum given a RMSF.
    """
    phiArr_radm2 = aDict["phiArr_radm2"]
    freqArr_Hz = aDict["freqArr_Hz"]
    weightArr = aDict["weightArr"]
    dirtyFDF = aDict["dirtyFDF"]  
    phi2Arr_radm2 = aDict["phiArr_radm2"]
    RMSFArr=aDict["RMSFArr"]


    lambdaSqArr_m2 = np.power(C/freqArr_Hz, 2.0)

    # If the cutoff is negative, assume it is a sigma level
    if verbose: print("Expected RMS noise = %.4g mJy/beam/rmsf", end=' ')
    (mDictS["dFDFth_Jybm"]*1e3)
    if cutoff<0:
        print("Using a sigma cutoff of %.1f.", end=' ')
        (-1 * cutoff),
        cutoff = -1 * mDictS["dFDFth_Jybm"] * cutoff
        print("Absolute value = %.3g", end=' ')
        cutoff
    else:
        print("Using an absolute cutoff of %.3g (%.1f x expected RMS).", end=' ')
        (cutoff, cutoff/mDictS["dFDFth_Jybm"])

    startTime = time.time()
    pdb.set_trace()    
    # Perform RM-clean on the spectrum
    cleanFDF, ccArr, iterCountArr = \
              do_rmclean_hogbom(dirtyFDF        = dirtyFDF,
                                phiArr_radm2    = phiArr_radm2,
                                RMSFArr         = RMSFArr,
                                phi2Arr_radm2   = phi2Arr_radm2,
                                fwhmRMSFArr     = np.array(mDictS["fwhmRMSF"]),
                                cutoff          = cutoff,
                                maxIter         = maxIter,
                                gain            = gain,
                                verbose         = False,
                                doPlots         = showPlots,
                                doAnimate       = doAnimate)
    cleanFDF #/= 1e3
    ccArr #/= 1e3

    # ALTERNATIVE RM_CLEAN CODE ----------------------------------------------#
    '''
    cleanFDF, ccArr, fwhmRMSF, iterCount = \
              do_rmclean(dirtyFDF     = dirtyFDF,
                         phiArr       = phiArr_radm2,
                         lamSqArr     = lamSqArr_m2,
                         cutoff       = cutoff,
                         maxIter      = maxIter,
                         gain         = gain,
                         weight       = weightArr,
                         RMSFArr      = RMSFArr,
                         RMSFphiArr   = phi2Arr_radm2,
                         fwhmRMSF     = mDictS["fwhmRMSF"],
                         doPlots      = True)
    '''
    #-------------------------------------------------------------------------#

    endTime = time.time()
    cputime = (endTime - startTime)
    print("> RM-CLEAN completed in %.4f seconds.", end=' ')
    cputime

    # Measure the parameters of the deconvolved FDF
    mDict = measure_FDF_parms(FDF         = cleanFDF,
                              phiArr      = phiArr_radm2,
                              fwhmRMSF    = mDictS["fwhmRMSF"],
                              #dFDF        = mDictS["dFDFth_Jybm"],
                              lamSqArr_m2 = lambdaSqArr_m2,
                              lam0Sq      = mDictS["lam0Sq_m2"])
    mDict["cleanCutoff"] = cutoff
    mDict["nIter"] = int(iterCountArr)

    # Measure the complexity of the clean component spectrum
    mDict["mom2CCFDF"] = measure_fdf_complexity(phiArr = phiArr_radm2,
                                                FDF = ccArr)
    
    # Save the deconvolved FDF and CC model to ASCII files
    print("Saving the clean FDF and component model to ASCII files.")
    outFile = prefixOut + "_FDFclean.dat"
    print("> %s", end=' ')
    outFile
    np.savetxt(outFile, zip(phiArr_radm2, cleanFDF.real, cleanFDF.imag))
    outFile = prefixOut + "_FDFmodel.dat"
    print("> %s", end=' ')
    outFile
    np.savetxt(outFile, zip(phiArr_radm2, ccArr))

    # Save the RM-clean measurements to a "key=value" text file
    print("Saving the measurements on the FDF in 'key=val' and JSON formats.")
    outFile = prefixOut + "_RMclean.dat"
    print("> %s", end=' ')
    outFile
    FH = open(outFile, "w")
    for k, v in mDict.iteritems():
        FH.write("%s=%s\n" % (k, v))
    FH.close()
    outFile = prefixOut + "_RMclean.json"
    print("> %s", end=' ')
    outFile
    json.dump(mDict, open(outFile, "w"))

    
    # Print the results to the screen
    print()
    print('-'*80)
    print('RESULTS:\n')
    print('FWHM RMSF = %.4g rad/m^2', end=' ')
    (mDictS["fwhmRMSF"])
    
    print('Pol Angle = %.4g (+/-%.4g) deg', end=' ')
    (mDict["polAngleFit_deg"],mDict["dPolAngleFit_deg"])
    print('Pol Angle 0 = %.4g (+/-%.4g) deg', end=' ')
    (mDict["polAngle0Fit_deg"],mDict["dPolAngle0Fit_deg"])
    print('Peak FD = %.4g (+/-%.4g) rad/m^2', end=' ')
    (mDict["phiPeakPIfit_rm2"],mDict["dPhiPeakPIfit_rm2"])
    print('freq0_GHz = %.4g ', end=' ')
    (mDictS["freq0_Hz"]/1e9)
    print('I freq0 = %.4g mJy/beam', end=' ')
    (mDictS["Ifreq0_mJybm"])
    print('Peak PI = %.4g (+/-%.4g) mJy/beam', end=' ')
    (mDict["ampPeakPIfit_Jybm"]*1e3,mDict["dAmpPeakPIfit_Jybm"]*1e3)
    print('QU Noise = %.4g mJy/beam', end=' ')
    (mDictS["dQU_Jybm"]*1e3)
    print('FDF Noise (measure) = %.4g mJy/beam', end=' ')
    (mDict["dFDFms_Jybm"]*1e3)
    print('FDF SNR = %.4g ', end=' ')
    (mDict["snrPIfit"])
    print()
    print('-'*80)
    
    # Pause to display the figure
    if showPlots or doAnimate:
        print("Press <RETURN> to exit ...", end=' ')
        raw_input()
        
    return mDict
        
def readFiles(fdfFile, rmsfFile, weightFile, rmSynthFile, nBits):

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)
        
    # Read the RMSF from the ASCII file
    phi2Arr_radm2, RMSFreal, RMSFimag = np.loadtxt(fdfFile, unpack=True, dtype=dtFloat)
    # Read the frequency vector for the lambda^2 array
    freqArr_Hz, weightArr = np.loadtxt(weightFile, unpack=True, dtype=dtFloat)
    # Read the FDF from the ASCII file
    phiArr_radm2, FDFreal, FDFimag = np.loadtxt(fdfFile, unpack=True, dtype=dtFloat)
    # Read the RM-synthesis parameters from the JSON file
    mDictS = json.load(open(rmSynthFile, "r"))
    dirtyFDF = FDFreal + 1j * FDFimag    
    RMSFArr = RMSFreal + 1j * RMSFimag
    
    #add array dictionary
    aDict = dict()
    aDict["phiArr_radm2"] = phiArr_radm2
    aDict["phi2Arr_radm2"] = phi2Arr_radm2
    aDict["RMSFArr"] = RMSFArr
    aDict["freqArr_Hz"] = freqArr_Hz
    aDict["weightArr"]=weightArr
    aDict["dirtyFDF"]=dirtyFDF
    
    return mDictS, aDict


