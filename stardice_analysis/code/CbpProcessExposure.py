"""
This file contains the function to process the CBP images, and a class to store the results in.
The function has to be outside of any classes in order to be parallelizable by python's internal
multiprocessing functions.

The ProcessedExposure class contains the results of the processed image, which will be used to generate a 
CBPScanResults object.
"""

import numpy as np
import astropy.io.fits as pft
import os
import CbpHelpers as cbph
import CbpDiagnostics as cbpd
import copy
import logging





class ExposureResults():
    def __init__(self):
        self.laser_wavelength = None
        self.image_number = None
        self.file_name = None
        self.exp_time = None
        self.filter = None
        self.charge = None
        self.charge_uncert = None
        self.spectrum = None
        self.n_spectra = None
        self.spec_saturated = None
        self.flux = None
        self.raw_flux = None
        self.flux_uncert = None
        self.dot_loc = None




def process_exposure(config, proc_image_metadata, image_num):
    """
    This function takes in scan paramters, relevant metadata, and a
    specific image number (corresponding to a specific image in the file_list).

    Essentially, this function handles *all* of the parallelizable tasks that 
    can be performed on a single CBP exposure.

    The Function:
    1. Reads relevant header data
    2. Subtracts bias and dark images
    3. Finds new spot locations (if enabled)
    4. Processes spectra (if enabled)
    5. Does aperture photometry
    6. Makes diagnostic plots (if enabled)

    pim: dictionary containing relevant metadata
    """
    # Convienience renaming
    pim = proc_image_metadata
    i = image_num 
    f = pim['file_list'][i]  # Which image is this thread working on?
    results = ExposureResults()

    # Load relevant metadata from fits header
    d = pft.open(f)
    wavelength = np.float(d[0].header['laserwavelength'])
    exp_time = np.float(d[0].header['EXPTIME'])
    # Sometimes the FILTER keyword doesn't exist...
    try:
        filt = np.int(d[0].header['FILTER'])
    except ValueError:
        filt = np.int(-99)

    # Fill in other results items that will be needed
    results.image_number = i
    results.file_name = f
    results.laser_wavelength = wavelength
    results.filter = filt
    results.exp_time = exp_time



    log_string = '%s %7.2f nm FILE: %d of %d' % (os.path.split(f)[-1], wavelength, (i+1)/2, len(pim['file_list'])/2) 
    logging.info(log_string)


    master_bias = cbph.MasterBias()
    bias = master_bias(d[0])
    if config.subtract_dark:
        dark_filename = pim['file_list'][i-1] # Always assume dark is the file immediately before
        dark_data = get_dark(dark_filename)   # get_dark provides a bias-subtracted dark image
        data = d[0].data.astype(np.float) - bias - dark_data
    else:
        dark_data = None
        data = d[0].data
        # must re-cast data as float before doing math, otherwise you'll get under/overflow
        # and have rounding troubles
        data = d[0].data.astype(np.float) - bias

    data *= config.gain

    results.charge = get_photodiode_charge(d, exp_time, config)

    # If we are allowed to move, but don't have initial mount pointings, we have to assume
    # that the spots are within a radius of spot_search_rad around the current location
    # use findCenter function to convolve and locate new centers
    if config.spots_can_move and not config.cbp_moves:
        new_dot_loc = cbph.find_center(data, pim['dot_loc'], search_radius=config.spot_search_rad, enforce_colinear=config.enforce_colinear)
    elif config.spots_can_move and config.cbp_moves:
        #  If we are allowed to move and cbp mount moves, check to see if we need to offset grid:
        shifts = {}
        new_dot_loc = copy.deepcopy(pim['dot_loc'])
        for k in pim['imps'].keys():
            if float(d[0].header[k]) != pim['imps'][k]:
                shifts[k] = float(d[0].header[k]) - pim['imps'][k]
        #  This will only trigger if there are >0 keys in shifts
        if shifts:
            sv = np.zeros((1, 2)) # sv  === shift vector
            for k in shifts.keys():
                sv += config.mount_derivatives[k] * shifts[k]
            sv = np.repeat(sv, len(new_dot_loc), axis=0)
            new_dot_loc += sv
    else:
        new_dot_loc = pim['dot_loc']

    results.dot_loc = new_dot_loc

    # Process Spectra
    # If process_spectra is False, results.<<spectra stuff>> all default to None
    if config.process_spectra:
        results.spectrum, results.n_spectra, results.spec_saturated = cbph.reduce_spectra(d[config.fits_spectrum_extname].data)
        if config.make_diagnostics:
            cbpd.make_spectrum_plot(config, results)

    #  =====================================================
    #  Aperture photometry
    phot_table, uncert = cbph.do_aperture_photometry(config, data, results.dot_loc)

    #  =====================================================
    results.flux = phot_table['residual_aperture_sum']
    results.raw_flux = phot_table['aperture_sum']
    results.flux_uncert = uncert

    #  ====================================================
    if config.make_diagnostics:
        cbpd.make_images(config, data, results, dark_data)

    return results

def get_dark(filename, master_bias=None):
    if master_bias is None:
        master_bias = cbph.MasterBias()
    dark = pft.open(filename)
    dark_data = dark[0].data
    dark_bias = master_bias(dark[0])
    return dark_data.astype(np.float) - dark_bias

def get_photodiode_charge(fits_header_list, exp_time, config):
    phd = fits_header_list[config.fits_pd_extname].data['phd']
    phd_time = fits_header_list[config.fits_pd_extname].data['time']
    if config.subtract_photodiode_bg:
        bkg_charge = cbph.estimate_charge_bkg(phd_time, phd, exp_time)
    else:
        bkg_charge = 0.
    return np.max(phd) - bkg_charge


