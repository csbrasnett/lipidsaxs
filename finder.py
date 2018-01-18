# -*- coding: utf-8 -*-
"""
@author: Chris Brasnett, University of Bristol, christopher.brasnett@bristol.ac.uk

This programme takes a file containing I(q) vs q data, and return an array of locations of Bragg peaks in the data.

It works by firstly finding an approximate location  of Bragg peaks in the data, eliminating peaks which fall outside of a
user-defined range, which will eliminate beamstop and detector noise at low and high values of q respectively. It then uses
the lmfit library to fit a convolution of a Voigt peak and a linear background to a range around the peak.

NB: The fitting range is hard-coded as the 7 points either side of the initially found peak in the discretised data, so care 
should be taken with regards to this fact if you are fitting data with much broader peaks. The programme will then remove peaks
which have been found multiple times by the inital search. If a figure of the q vs. I(q) data overlaid with lines where the 
peaks have been fitted to in q is wanted, this can then be displayed if wanted - defined by one of the programme parameters.

pass the following parameters to this function:
    file_name - the full file path to the I vs q data that you want to find Bragg peaks in. Must be formatted q in first (0)
                column, I(q) in second (1) column
    
    finding_sensitivity - how sensitive the programme needs to be to finding peaks. The lower the number is the more sensitive
                          the programme will be. If there is a lower background, this value can be quite high (ie >40). For 
                          noisier data, ~8 will suffice.
                         
    lo_lim, hi_lim - these are parameters specifying the low and high limits of the q range of where the peaks can be found.
    
    fig - set this to 1 if you would like a figure returned of the I vs q data on a log(I) plot complete with overlaid lines
          of where the peaks have been found - and returned.
"""

import lmfit as lm
import numpy as np
import matplotlib.pyplot as plt
from detect_peaks import detect_peaks

def fitting(x,y,approx_centre,plot):
    #fit the peak using a convolution of an exponential function and a Voigt peak
    lin_mod = lm.models.LinearModel(prefix='lin_')
    pars = lin_mod.guess(y, x=x)

    Voigt_model=lm.models.VoigtModel(prefix='V_')
    pars.update(Voigt_model.make_params())
    
    '''
    define the inital Voigt variables: centre as the centre found in the data so far, the width as the width of the 
    window in which the peak has been defined, and the amplitude as the width of the intensity of the window in which 
    the fitting is being done. NB: free variation of the gamma parameter of the Voigt model does not work well.
    '''
    mod = Voigt_model  + lin_mod
            
    pars['V_center'].set(approx_centre)
    pars['V_sigma'].set((np.max(x)-np.min(x))/5)
    pars['V_amplitude'].set((np.max(y)-np.min(y))/50)
    #pars['V_gamma'].set(vary=True)
    
    #do the fitting
    result=mod.fit(y,pars,x=x)
    
    #in case you want to look at the fit in each case
    if plot==1:
        comps=result.eval_components(x=np.arange(x[0],x[-1],0.0001))
        plt.plot(x,y,'go')
        plt.plot(x,result.best_fit, 'r')
        plt.axvline(approx_centre,c='g')
        plt.axvline(result.params['V_center'].value,c='b')
        plt.plot(np.arange(x[0],x[-1],0.0001),comps['lin_'],'b--')
        plt.plot(np.arange(x[0],x[-1],0.0001),comps['V_'],'b--')
        plt.show()
        plt.clf()
        print(result.fit_report())
      
    fitted_centre=result.params['V_center'].value

    return fitted_centre,result.redchi


def finder(file_name,lo_lim,hi_lim,fig):
    table=np.genfromtxt(file_name,delimiter='\t')
    x=table[0:,0]
    y=table[0:,1]
    
    #have a first go at finding the approximate position of the peaks in the data
    ind=detect_peaks(y,mpd=10)
    
    #limit the low and high q noise; return the peaks' positions as an element index in the I array 
    second_pass_peaks=np.zeros(0)
    for i in range(0,len(ind)):
        if x[ind[i]]>lo_lim and x[ind[i]]<hi_lim:
            second_pass_peaks=np.append(second_pass_peaks,ind[i])
    second_pass_peaks=np.trim_zeros(second_pass_peaks.astype(dtype=int),'f')


    #now use lmfit to try to fit the peak properly
    fitted_centre=np.zeros(0)
    for i in range(0,len(second_pass_peaks)):       
        #give a range over which to fit the data from the integer indexed found peak in the data  
        fit_range=7
        x_range=x[int(second_pass_peaks[i])-fit_range:int(second_pass_peaks[i])+fit_range]
        y_range=y[int(second_pass_peaks[i])-fit_range:int(second_pass_peaks[i])+fit_range]
        
        #pass the data to fit to the fitting function.
        fit=fitting(x_range,y_range,x[int(second_pass_peaks[i])],0)
    
        fitted_centre=np.append(fitted_centre,fit)
    
    #filter the fitted peaks again to ensure that they're within the correct search range.
    returning_peaks=np.zeros(0)
    for i in range(0,len(fitted_centre)):
        if fitted_centre[i]>lo_lim and fitted_centre[i]<hi_lim:
            returning_peaks=np.append(returning_peaks,fitted_centre[i])

    #plot the data with lines through the peaks found if you want.
    if fig==1:    
        plt.semilogy(x,y)
        plt.xlim(lo_lim,hi_lim)
        plt.xlabel('q (Å$^{-1}$)')
        plt.ylabel('Intensity (A.U.)')
        for k in range(0,len(returning_peaks)):
            plt.axvline(returning_peaks[k],c='g')
        plt.show()
    
    return returning_peaks