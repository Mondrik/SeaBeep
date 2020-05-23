
"""
This class holds the configuration paramters for a particular CBP processing run.
By default, it is pre-loaded with "reasonable" values for all relevant paramters, 
but you might want to change some of them depending on what you're doing.

Nick Mondrik, May 2020
"""

import pickle
import logging



class CBPConfig():
    def __init__(self):
        # Name of log file. If the name is '', no log file will be created.
        self.log_file_name = ''

        # Locations of data and other utility files needed to process the data
        
        # FOLDER holding data to be processed
        self.data_location = ''

        # File that contains information on how spot xy detector location changes w.r.t. mount alt/az
        self.mount_derivatives = pickle.load(open('../data/derivs.pkl','rb'))
        
        # Pixel-wavelength map for spectrograph
        self.pix2wave_fit_file = '../data/pix2wave_chebfit.txt'


        # Aperture photometry configuration parameters
        self.ap_phot_rad = 45                    # Aperture photometry radius
        self.sky_rad_in = self.ap_phot_rad + 10  # Inner radius of sky annulus
        self.sky_rad_out = self.sky_rad_in + 5   # Outer radius of sky annulus
        self.read_noise = 16.                    # e- RMS read noise
        self.gain = 1.                           # gain in ADU/e-        

        # Pre-processing configuration parameters
        self.subtract_dark = True                # Do we subtract dark frames?
        self.spots_can_move = True               # Do we allow spots to move (as a grid)?
        self.enforce_colinear = True             # If dots can move, do they have to move as a grid?
        self.spot_search_rad = 10                # How far away do we look for spots?
        self.process_spectra = True              # Do we try to reduce spectra?
        self.cbp_moves = False                   # Are we running a scan where we moved the CBP mount?
        self.subtract_photodiode_bg = True       # Do we subtract of the estimated photodiode background charge?
        
        # Data processing configuration parameters
        self.fits_pd_extname = 'PHOTOCOUNT'      # Extension name for photodiode data in fits file
        self.fits_spectrum_extname = 'SPECTRA'   # Extension name for spectrometer data in fits file
        self.n_prepost_pd_reads = 10             # Number of photodiode charge reads before/after exposure
        self.spec_fit_region_size = 20           # Size of region around peak to use in wavelength extraction
        self.wavelength_interp_spline_deg = 2    # Degree of wavelength interpolating spline

        
    
