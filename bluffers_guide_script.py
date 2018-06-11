# -*- coding: utf-8 -*-
"""

author: Chris Brasnett, University of Bristol, christopher.brasnett@bristol.ac.uk

"""

import sys
import glob
import os
import numpy as np

'''
begin editable section
'''

#where is the lipidsaxs folder
path_to_lipidsaxs='your/folder/path/here/'

#folder containing csvs of I vs q data
data_folder='your/data/path/here/'

#folder to save peak overlaid figures to
fig_save_dir=os.path.dirname(os.path.realpath(__file__))

#save the phase identification results as a text file in this directory
text_save_dir=os.path.dirname(os.path.realpath(__file__))

#some code parameters as explained in the documentation

#low q value to search for peaks in 
low_q=0.04

#high q value to search for peaks in
high_q=0.35

#the minimum separation of the peaks
peak_separation=0.01

#do you want to save the overlaid figures? True if yes, False if not.
save_figures=True

#show the plots being saved in the IDE as you go along?
in_IDE_plots=True

#which instrument was the data measured on? Give 'Ganesha' or 'DLS' (in quotes) for Ganesha or Diamond respectively.
instrument='Ganesha'

'''
end editable section
'''

sys.path.append(path_to_lipidsaxs)
import lipidsaxs

files=glob.glob(data_folder+'*.csv')

if instrument=='Ganesha':
    instrument_switch_Ganesha=True
    instrument_switch_DLS=False
elif instrument=='DLS':
    instrument_switch_Ganesha=False
    instrument_switch_DLS=True
else:
    print('Instrument variable has not been set correctly!')

print("\rThe numbering of the files will be saved as %s onwards. If this doesn't make sense, stop the programme now!" %files[0][-21:-16])

p=1
for i in files:
    print('Progress: %d/%d' %(p,len(files)))
    test=lipidsaxs.finder(i,low_q,high_q,peak_separation,Ganesha=instrument_switch_Ganesha,DLS=instrument_switch_DLS,plot=in_IDE_plots,savefig=save_figures,savedir=fig_save_dir)
    
    #change ordering here so that every file title is written, then phase is tested, then write phase info to file
    #if no phase info, then write 'none' or something
    if type(test)!=int:
        phase=lipidsaxs.main(test,low_q)
        with open(text_save_dir+'output.txt', 'a') as f:
            name=i[-21:-16]
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
    p=p+1