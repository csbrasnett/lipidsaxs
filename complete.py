# -*- coding: utf-8 -*-
"""

author: Chris Brasnett, University of Bristol, christopher.brasnett@bristol.ac.uk

"""

import glob
from natsort import natsorted
from phase_ID import main
import finder

def complete(folder):
    files=natsorted(glob.glob(folder+'/*.csv'))
    print(files)
    low_q=0.09
    high_q=0.25
    minimum_separation=0.001
    
    bin_threshold=3
    Q_bin_width_factor=0.1
    La_HII_bin_width_factor=2
    
    for i in range(0,len(files)):
        print(files[i])
        peaks=finder.finder(files[i],low_q,high_q,minimum_separation)        
        phase_dict=main(peaks,low_q,bin_threshold,Q_bin_width_factor,La_HII_bin_width_factor)
    
    return peaks,phase_dict

test=complete('your/text/folder/path/here')
print(test)