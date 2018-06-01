# -*- coding: utf-8 -*-
"""
This programme takes a file containing I(q) vs q data (in 0th and 1st columns respectively of a text file) and return an array of locations of Bragg peaks in the data.

author: Chris Brasnett, University of Bristol, christopher.brasnett@bristol.ac.uk


It works by fitting a convolution of a Voigt peak and a linear background to a continous window of data throughout a given q range.
Refinement then happens to clarify the true positions of peaks in the

NB: The fitting range is hard-coded as a 15 point window in the discretised data, so care 
should be taken with regards to this fact if you are fitting data with much broader peaks. The programme will then remove peaks
which have been found multiple times by the inital search. If a figure of the q vs. I(q) data overlaid with lines where the 
peaks have been fitted to in q is wanted, this can then be displayed if wanted - defined by one of the programme parameters.

pass the following parameters to this function:
    file_name - the full file path to the I vs q data that you want to find Bragg peaks in. Must be formatted q in first (0)
                column, I(q) in second (1) column
    
    finding_sensitivity - how sensitive the programme needs to be to finding peaks. The lower the number is the more sensitive
                          the programme will be. If there is a lower background, this value can be quite high (ie >40). For 
                          noisier data, ~8 will suffice.
                         
    lower_limit, upper_limit - these are parameters specifying the low and high limits of the q range of where the peaks can be found.
    
    min_sep - the minimum separation for the peaks in q.
    
    Ganesha, DLS - optional in name, but not in practice. Where the data was taken (in house or at Diamond) will affect how some of 
                    the refinement routines behave. See the extended documentation for more details.
     
    fig - optional, set as True if you want to see a figure of the peaks found overlaid on the data passed to the function.
"""

import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm

def fitting(x,y,approx_centre,height_threshold,plot=False):
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
    
    fitted_centre=result.params['V_center'].value
    sigma=result.params['V_sigma'].value    
    height=result.params['V_amplitude'].value
    
    #eliminate terrible fits
    if height>height_threshold and fitted_centre<max(x) and fitted_centre>min(x):
        #in case you want to look at the fit in each case
        if plot==True:
            comps=result.eval_components(x=np.arange(x[0],x[-1],0.0001))
            plt.plot(x,y,'go',label='data')
            plt.plot(x,result.best_fit, 'r',label='result fit to data')
            plt.axvline(approx_centre,c='g',label='initial centre')
            plt.axvline(result.params['V_center'].value,c='b',label='peak centre')
            plt.plot(np.arange(x[0],x[-1],0.0001),comps['lin_'],'b--',label='linear component of fit')
            plt.plot(np.arange(x[0],x[-1],0.0001),comps['V_'],'--',label='Voigt peak component of fit')
            plt.legend()
            plt.xlabel('q (Å$^{-1}$)')
            plt.ylabel('Intensity (A.U.)')
            plt.show()
            plt.clf()
            #print(result.fit_report())
            print(fitted_centre,sigma,height)
        #return the results that meet the conditions
        return fitted_centre,sigma,height
    else: return 0

def finder(file_name,lower_limit,upper_limit,min_sep, Ganesha=False,DLS=False,plot=False):
    
    try:
        if Ganesha==True:
            delim_str=','
            ht_threshold=0.0001
            
        if DLS==True:
            delim_str='\t'
            ht_threshold=0.1
            
        #get the data from the file
        table=np.genfromtxt(file_name,delimiter=delim_str,skip_header=10)
        
        #cut out the x and y data defined by the q range.
        x_data=table[np.intersect1d(np.where(table[0:,0]>lower_limit),np.where(table[0:,0]<upper_limit)),0]
        y_data=table[np.intersect1d(np.where(table[0:,0]>lower_limit),np.where(table[0:,0]<upper_limit)),1]
    
        #the number of data points to trial fits across
        fitting_range=10
        
        #attempt to fit the data across a moving window of the q range of interest. This will find peaks multiple times over.
        peaks=np.zeros(0)
        for i in range(0,(np.where(table[0:,0]<upper_limit)[-1][-1]-np.where(table[0:,0]>lower_limit)[0][0]-fitting_range)):
            x=x_data[i:(i+fitting_range)]
            y=y_data[i:(i+fitting_range)]
            
            result=fitting(x,y,np.mean(x),height_threshold=ht_threshold)
            
            if result != 0:
                peaks=np.append(peaks, result[0])
              
        #define the minimum separation between peaks - otherwise the binning of the data will put separate peaks into one bin.
        #bin the peaks found during the fitting procedure
        hist, bin_edges=np.histogram(peaks,bins=np.arange(min(peaks), max(peaks) + min_sep, min_sep))
        inds=np.digitize(peaks,bin_edges)
        
        returning_peaks=np.zeros(0)
        for i in range(0, np.size(np.arange(min(peaks), max(peaks) + min_sep, min_sep))):
            try:
                #look forwards and backward to catch each bin incase the values have leaked between boundaries
                previous_bin=peaks[np.where(inds==(i-1))]
                this_bin=peaks[np.where(inds==i)]
                next_bin=peaks[np.where(inds==(i+1))]
                
                #if two bins are next to each other, group them together and average those values to return
                if len(this_bin)>0 and len(previous_bin)>0 and len(next_bin)==0:
                    conc_bin=np.concatenate((this_bin,previous_bin))
                    returning_peaks=np.append(returning_peaks,np.mean(conc_bin))
                
                #otherwise just average the bin and return it as the peak.
                elif len(this_bin)>0 and len(previous_bin)==0 and len(next_bin)==0:
                    returning_peaks=np.append(returning_peaks,np.mean(this_bin))
                    
            except IndexError:
                pass
            
        if plot==True:
            plt.plot(x_data,y_data)
            for i in returning_peaks:
                plt.axvline(i,c='r')
            plt.xlabel('q (Å$^{-1}$)')
            plt.ylabel('Intensity (A.U.)')
            plt.show()
            plt.clf()
            
        return returning_peaks

    except UnboundLocalError:
        print('Error! You must tell the programme where the data was collected in order to use the peak finder.')
