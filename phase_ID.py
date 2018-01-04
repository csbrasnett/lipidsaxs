# -*- coding: utf-8 -*-
"""
@author: Chris Brasnett, University of Bristol, christopher.brasnett@bristol.ac.uk

This programme will take an input array of peaks in 1D I vs q data (such as those returned from the finder programme),
and returns a dictionary of possible phases that the data can take on, along with the miller plane index and the peaks 
used to for that possible phase assignment. Also needs to be passed is a threshold for the number of binned values in 
the histogram. A value of around 8 should suffice for this purpose.
"""



"""
possible_phases works by creating matrices of lattice parameter values that can arise having declared that any peak that 
has been found can be indexed as any miller index for any phase. These values are then collapsed into a single 1D array,
which is investigated as a histogram. The number of bins in teh histogram is arbitrarily taken as twice the number of values,
so care should taken. Peaks in the histogram will arise at the points where there are matching values 
resulting from peaks being correctly indexed in the correct phase. The possible_phases takes a threshold number, such that 
bins with more values in it than the threshold are considered to be possible phase values. This is due to the fact
that because of symmetry degeneracies, 'correct' phase values may arise from more than a single phase matrix. The values
in the bins which exceed threshold population are then investigated for their origins: which peak and index were 
responsible for bringing them about? 

The possible_phases will return a dictionary, keyed through lattice parameters, with associated values of the phase (D=0, P=1, G=3),
the peaks that have been indexed, and the indicies assigned to the peak.

pass the following parameters to this function:
    peaks - an array of peaks that have previously been found elsewhere
    
    bin_factor - a value to choose the number of bins. The number of bins in the histogram is calculated as the product of the 
                 bin_factor value and the length of the flattened matrix of possible lattice parameter values. Take care such
                 that this value must make the bins values an integer.
    
    threshold - a small number determining the threshold for filtering the bin population histogram when searching for matching 
                values in the peak assignment histogram.


"""

import numpy as np

def possible_phases(peaks,bin_factor, threshold):
        
    #define the characteristic peak ratios
    QIID=np.array([2,3,4,6,8,9,10,12])[:,np.newaxis]
    QIIP=np.array([2,4,6,8,10,12,14,16])[:,np.newaxis]
    QIIG=np.array([6,8,14,16,20,22,24,26])[:,np.newaxis]
    
    QIID_ratios=np.sqrt(QIID)
    QIIP_ratios=np.sqrt(QIIP)
    QIIG_ratios=np.sqrt(QIIG)
    
    #1) create matrices of all possible lattice parameter values
    #2) flatten each matrix to one dimension
    #3) combine the matricies into one
    D_init = 2*np.pi*(1/peaks)*QIID_ratios
    P_init = 2*np.pi*(1/peaks)*QIIP_ratios
    G_init = 2*np.pi*(1/peaks)*QIIG_ratios
    
    n=np.reshape(np.arange(0,np.size(D_init)),np.shape(D_init))
    
    D=np.ndarray.flatten(D_init)
    P=np.ndarray.flatten(P_init)
    G=np.ndarray.flatten(G_init)
    
    values=np.concatenate((D,P,G))
    
    #histogram the data so that we have some bins
    hist, bin_edges=np.histogram(values,bins=bin_factor*np.size(values))
    
    #digitise the data (see numpy docs for explanations)
    inds=np.digitize(values,bin_edges)
    
    #will return the possible phases, their lattice parameters, and the peaks and hkl index from which they arise as a dictionary.
    phase_dict={}

    for i in range(0, np.size(values)):
        try:
            #find the values from the values array which are actually present in each bin and put them in the values array
            binned_values=values[np.where(inds==i)]
            #this size filtering is completely arbitrary. 
            if np.size(binned_values)>threshold:                
                #trace where the values in the bin originated from in the arrays.
                positions_array=np.zeros(0)
                for k in range(0, np.size(binned_values)):
                    positions_array=np.append(positions_array,np.where(binned_values[k]==values)[0])
                
                #look at the distribution of the origin of the arrays - they should be group dependent on the phase.
                #D_sourced, P_sourced, G_sourced are the positions in the values array where the matching peaks have come from
                final_pos_array=np.unique(positions_array)
    
                #split the positions up into which cubic phase calculation they have come from.         
                D_factors=np.where(final_pos_array<len(D))[0][0:]
                P_factors=(np.where(final_pos_array<=(len(P)+len(D))-1)[0][0:])[len(D_factors):]
                G_factors=np.where(final_pos_array>(len(P)+len(D))-1)[0][0:]
    
                #correspond the positions in the factors arrays to where they come from in the final positions array            
                D_sourced=final_pos_array[D_factors].astype(int)
                P_sourced=final_pos_array[P_factors].astype(int)
                G_sourced=final_pos_array[G_factors].astype(int)
                
                #want to find where the matching phases have come from in the array to see which one is the real one.
                #
                #np.mod(o_sourced[a],n) corrects the position in the o array for running the same length as the sourced array
                #then find where the value is the same to identify the row
                #then find from which ratio factor the peak originated from.         
                D_sourced_factors=np.zeros(0,dtype=np.int)
                P_sourced_factors=np.zeros(0,dtype=np.int)
                G_sourced_factors=np.zeros(0,dtype=np.int)
                
                D_sourced_peaks=np.zeros(0)
                P_sourced_peaks=np.zeros(0)
                G_sourced_peaks=np.zeros(0)
                
                for a in range(0,len(D_sourced)):
                    D_hkl=QIID[np.where(np.mod(D_sourced[a],np.size(n))==n)[0]][0][0]
                    D_peak_hkl=peaks[np.where(np.mod(D_sourced[a],np.size(n))==n)[1]][0]
                    
                    D_sourced_factors=np.append(D_sourced_factors,np.int(D_hkl))
                    D_sourced_peaks=np.append(D_sourced_peaks,D_peak_hkl)
                    
                #only save the phase (as number: D=0, P=1,G=2), and related data to the returned dictionary if there are more than 3 peaks in there.
                if len(D_sourced_factors) >3:
                    phase_dict[values[D_sourced[a]]] = 0, D_sourced_factors, D_sourced_peaks
                    
                for b in range(0,len(P_sourced)):
                    P_hkl=QIIP[np.where(np.mod(P_sourced[b],np.size(n))==n)[0]][0][0]
                    P_peak_hkl=peaks[np.where(np.mod(P_sourced[b],np.size(n))==n)[1]][0]
                    
                    P_sourced_factors=np.append(P_sourced_factors,P_hkl)
                    P_sourced_peaks=np.append(P_sourced_peaks,P_peak_hkl)
                if len(P_sourced_factors) >3:
                    phase_dict[values[P_sourced[a]]] = 1, P_sourced_factors, P_sourced_peaks
                    
                for c in range(0,len(G_sourced)):
                    G_hkl=QIIG[np.where(np.mod(G_sourced[c],np.size(n))==n)[0]][0][0]
                    G_peak_hkl=peaks[np.where(np.mod(G_sourced[c],np.size(n))==n)[1]][0]
                    
                    G_sourced_factors=np.append(G_sourced_factors,G_hkl)
                    G_sourced_peaks=np.append(G_sourced_peaks,G_peak_hkl)
                if len(G_sourced_factors) >3:
                    phase_dict[values[G_sourced[a]]] = 2, G_sourced_factors, G_sourced_peaks
               
        except IndexError:
            pass
    return phase_dict


"""
projection_testing is the final clarification stage of identifying which of the possible identified phases are 'real'.
The phases are checked against a fundamental 'mode' that the lattice parameter and phase identified. From this fundamental
value, the peaks in q which should exist can be calculated. These proposed peaks are subsequently checked against the peaks
which actually exist in the data. This is done through constructing a difference matrix, populated by the differences between
the peaks in the projected and physical arrays. The matrix is then searched for where the value is very small - ie. the proposed
peak is present in the physical data. If all or all but one or two of the proposed peaks are present in the physical data, 
then it is said that that phase proposed is real, and not a feature of degenerate symmetry in the data.

pass the following parameters to this function:
    
    phase_array - the integer spacing ratios of the proposed phase that needs to be tested.
    
    fundamental - the ratio of a peak value of a phase to the square root of its index. Defined in the main below as the average
                  of these values across a set of peaks in a proposed phase.
    
    peak_array  - the full set of peaks that have been actually been physically found in the data, to test against a set of peaks
                  which should exist given the peaks present.
                  
    lo_q      - the same low limit in q that was used to define the width in which peaks are to be found

"""

def projection_testing(phase_array, fundamental, peak_array,lo_q):
    #now project the fundamental q value over the phase
    projected_values=(np.sqrt(phase_array)*fundamental)[:,np.newaxis]
    #check that the first projected peak is within the finding q width:
    if projected_values[0]>lo_q:

        #create a difference matrix, examining the difference in the proposed and existant peaks.     
        matching=np.abs(np.subtract(projected_values,peak_array))
      
        #the matches vavriable is an evaluation of where peaks that have been projected correspond to peaks that actually exist.
        #arbitrarily, if the difference in the lengths of the arrays is less than 2, (Ie. all peaks are present or only one or two 
        #are missing in the data) then return a confirmation that the phase is a real assignment of the peaks.
        matches=np.where(matching<0.001)[0]
        #matches1=np.where(matching<0.001)[1]

        if np.abs(len(projected_values)-len(projected_values[np.unique(matches)]))<2:
            return 1
    #if the lowest peak is not in the desired q range
    else:
        return 0
     

"""
the main module runs the above modules, passing the required data from one to the other.

pass the following parameters to this function:
    peaks - an array of peaks that have previously been found elsewhere
    
    bin_factor - a value to choose the number of bins. The number of bins in the histogram is calculated as the product of the 
                 bin_factor value and the length of the flattened matrix of possible lattice parameter values. Take care such
                 that this value must make the bins values an integer.
    
    threshold - a small number determining the threshold for filtering the bin population histogram when searching for matching 
                values in the peak assignment histogram.
    
    lo_q      - the same low limit in q that was used to define the width in which peaks are to be found

"""
def main(peaks,bin_factor,threshold,lo_q):

    QIID_ratios=np.array([2,3,4,6,8,9])
    QIIP_ratios=np.array([2,4,6,8,10,12])
    QIIG_ratios=np.array([6,8,14,16,20,22])

    phases=possible_phases(peaks,1,threshold)
    clar={}
    for value in phases.items():
        fundamental=np.mean(value[1][2]/np.sqrt(value[1][1]))
        if value[1][0]==0:
            test1=projection_testing(QIID_ratios,fundamental,peaks,lo_q)
            if test1==1:
                clar['D']=value[0],value[1][1],value[1][2]
        if value[1][0]==1:
            test2=projection_testing(QIIP_ratios,fundamental,peaks,lo_q)
            if test2==1:
                clar['P']=value[0],value[1][1],value[1][2]
        if value[1][0]==2:
            test3=projection_testing(QIIG_ratios,fundamental,peaks,lo_q)
            if test3==1:
                clar['G']=value[0],value[1][1],value[1][2]
    return clar
