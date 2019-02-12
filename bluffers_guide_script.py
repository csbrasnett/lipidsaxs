# -*- coding: utf-8 -*-
"""

author: Chris Brasnett, University of Bristol, christopher.brasnett@bristol.ac.uk

"""

import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def without_invalid(d):
    keys = {'unassigned_peaks'}
    return {x: d[x] for x in d if x not in keys}

'''
begin editable section
'''

#where is the lipidsaxs folder
path_to_lipidsaxs='your/folder/path/here/'

#folder containing csvs of I vs q data
data_folder='your/data/path/here/'

file_extensions='.csv'

#some code parameters as explained in the documentation

#low q value to search for peaks in 
low_q=0.04

#high q value to search for peaks in
high_q=0.35

#do you want to save the overlaid figures? True if yes, False if not.
save_figures=True

#folder to save peak overlaid figures to
fig_save_dir=os.path.dirname(os.path.realpath(__file__))

#save the phase identification results as a text file in this directory
text_save_dir=os.path.dirname(os.path.realpath(__file__))

#show the plots being saved in the IDE as you go along?
in_IDE_plots=True

#which instrument was the data measured on? Give 'Ganesha' or 'DLS' (in quotes) for Ganesha or Diamond respectively.
instrument='Ganesha'

#if you can see peaks that aren't being picked up, try changing this value arouond a bit. THe defaults are 0.1 for Diamond data and 0.0001 for Ganesha
peak_heights = None

'''
end editable section
'''

sys.path.append(path_to_lipidsaxs)
import lipidsaxs

files=glob.glob(data_folder+'*'+file_extensions)

if instrument=='Ganesha':
    instrument_switch_Ganesha=True
    instrument_switch_DLS=False
elif instrument=='DLS':
    instrument_switch_Ganesha=False
    instrument_switch_DLS=True
else:
    print('Instrument variable has not been set correctly!')

#this should find the correct code for the file. It splits the first file name by the directory folder, takes the last split, and then uses the code.
#the 1: might need to be changed to 0:, but -4 should get rid of the file label (.csv/.txt)
splitting = files[0].split(data_folder[:-1])[-1][1:-4]

print("\rThe numbering of the files will be saved as %s onwards. If this doesn't make sense, stop the programme now, and consider how to use the splitting variable!" %splitting)

p=1
for i in files[5:]:
    print('Progress: %d/%d' %(p,len(files)))
    peaks,saxs_data_x,saxs_data_y=lipidsaxs.finder(i,low_q,high_q,Ganesha=instrument_switch_Ganesha,DLS=instrument_switch_DLS,plot=in_IDE_plots,savefig=save_figures,savedir=fig_save_dir,ht_thresh=peak_heights)
    
    #change ordering here so that every file title is written, then phase is tested, then write phase info to file
    #if no phase info, then write 'none' or something
    if type(peaks)!=int:
        phase=lipidsaxs.main(peaks,low_q)
        
        #plot the data
        if in_IDE_plots==True:
            #exclude unassigned peaks from plot
            plotting_phases=without_invalid(phase)
            plt.semilogy(saxs_data_x,saxs_data_y)
            a=0
            for key in plotting_phases.keys():
                if key == 'D' or key=='G' or key=='P':
                    plt.text(low_q,saxs_data_y.mean()*((1+a)*10),key)
                    for j in range(len(phase[key][1])):
                        plt.axvline(plotting_phases[key][2][j],c='r')
                        plt.text(plotting_phases[key][2][j],saxs_data_y.mean()*((1+a)*10),str(plotting_phases[key][1][j]))
                a=a+1
            plt.xlabel('$q$ (Å$^{-1}$)')
            plt.ylabel('Intensity (A.U.)')
            if save_figures==True:
                name=i.split('\\')[-1][:-4]
                plt.savefig(fig_save_dir+'/'+name+'_phases.png',dpi=200)
            plt.show()
            plt.clf()
        
        with open(text_save_dir+'/output.txt', 'a') as f:
            name=i.split(data_folder[:-1])[-1][1:-4]
            f.write(name+':\n')
            for key in phase.keys():                
                f.write('%s\t' %key)
                for j in phase[key]:
                    if type(j)==np.ndarray:
                        for k in j:
                            f.write('%f\t' %k)
                    else:
                        f.write('%f\t' %j)
                f.write('\n')
            f.write('\n')
    elif type(peaks)==int:
        with open(text_save_dir+'/output.txt', 'a') as f:
            name=i.split(data_folder[:-1])[-1][1:-4]
            f.write(name+':\n')
            f.write('no peaks found')
            f.write('\n')            
    p=p+1