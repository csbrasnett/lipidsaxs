# lipidsaxs
tools for the analysis of lipid mesophase saxs data

This is a repository for scripts written in python for the analysis of lipid mesophase small-angle X-ray scattering data. The two key scripts are finder.py and phase_id.py. For an excellent introduction to the field of lipid mesophases, as well as an introduction to the analysis of mesophase saxs data, see Kulkarni et al. Phys. Chem. Chem. Phys.,13, 3004-3021 (2011).

The methodology used in the scripts is well commented.

finder.py will attempt to find mesophase Bragg peaks in 1D (I vs. q), and will return a numpy array of the peaks.

phase_ID.py will attempt to identify the cubic mesophase of a set of Bragg peaks given to it.
