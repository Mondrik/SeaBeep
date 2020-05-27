"""
This file contains a class to manage the processing of a CBP scan.
Scan parameters are defined in a CbpConfig object, which must be passed to the
CbpScan class at the time of creation.

Also contains the Dot class, which stores the by-spot relevant results for each pinhole spot
"""

import time
import os
import glob
import numpy as np
import pickle
import astropy.io.fits as pft
import multiprocessing as mp
from functools import partial
from CbpProcessExposure import process_exposure
import CbpHelpers as cbph
import CbpCalib as cc
from scipy.interpolate import UnivariateSpline
import logging
import sys



class CbpScan():
    def __init__(self, config, load_from_dict=None):
        """
        Class to process scans and hold their results.
        The results of a scan can be stored in a file by calling the 
        CbpScan.save(my_filename) method.

        The results can be loaded from disk by calling CbpScan.load(my_filename).
        Note that you DO NOT need to instantiate a new class first -- i.e.,
        my_old_scan = cs.CbpScan.load(my_filename) is *all* that is required to load
        an old scan.
        """
        if load_from_dict is not None:
            self.__dict__.clear()
            self.__dict__.update(load_from_dict)
            return

        self.config = config
        if self.config.log_file_name:
            handlers = [logging.FileHandler(self.config.log_file_name, mode='w'),
                        logging.StreamHandler(sys.stdout)]
            handlers[0].setLevel(logging.DEBUG)
            handlers[1].setLevel(logging.INFO)
            fmt = logging.Formatter("%(asctime)s:[%(levelname)-5.5s]:[%(funcName)s]:%(message)s")
            for h in handlers: h.setFormatter(fmt)
            logging.basicConfig(handlers=handlers, level=logging.DEBUG)
            logging.info('Log file %s created' % self.config.log_file_name)
        self.base_name = os.path.split(self.config.data_location)[-1]
        self.pim = self._load_process_image_metadata()

        # First, we need to set up all of the metadata and information that needs to be passed to
        # the processExposure function

        # We assume image 0 is a dark, image 1 is a light, image 2 is a dark, etc.
        # This gives us the indexes of the light images
        self.image_numbers = np.arange(0, len(self.pim['file_list']), 1)[1::2]
        self.n_images = self.image_numbers.size
       
        # Each dot on the focal plane is stored as a Dot object
        # Data for each dot will be loaded into the appropriate object at the end of the scan
        self.dots = [Dot(i, self.n_images) for i in range(self.pim['dot_loc'].shape[0])]

        # Initialize storage for exposure-level output quantities
        # These are the quantities that are unique to each exposure
        # and are thus *shared* across all dots.
        self.file_name = np.zeros(self.n_images).astype(np.str)
        self.dark_file_name = np.zeros(self.n_images).astype(np.str)
        self.filter = np.zeros(self.n_images).astype(np.int)
        self.laser_wavelength = np.zeros(self.n_images).astype(np.float)
        self.measured_wavelength = np.zeros(self.n_images).astype(np.float)
        self.wavelength_splinefit = np.zeros(self.n_images).astype(np.float)
        self.exp_time = np.zeros(self.n_images).astype(np.float)
        self.charge = np.zeros(self.n_images).astype(np.float)
        self.charge_uncert = np.zeros(self.n_images).astype(np.float)
        self.cbp_transmission = np.zeros(self.n_images).astype(np.float)
        self.cbp_uncert = np.zeros(self.n_images).astype(np.float)
        # Magic numbers are from the shape of the spectrograph data
        self.spectrum = np.zeros((self.n_images, 994, 2)).astype(np.float)
        self.n_spectra = np.zeros(self.n_images).astype(np.int)
        self.spec_saturated = np.zeros(self.n_images).astype(np.bool)

        self.wavelength_interp_spline = None
        



    def _load_process_image_metadata(self):
        """
        This is the metadata that each thread needs to analyze it's given image
        (for multiprocessing)
        """
        file_list = self._get_file_list()
        logging.info('Found %d fits files in data directory' % len(file_list))
        imps = self._get_imps(file_list[0])
        dot_loc = self._get_dots()
        logging.info('Found positions for %d dots' % dot_loc.shape[0])
        pim = {'file_list': file_list, 'imps': imps, 'dot_loc': dot_loc}
        return pim

    def _get_imps(self, test_file):
        """
        IMPs = Initial Mount Pointings
        They are used in conjunction with mount derivatives to calculate the focal plane
        offset introduced by rotating the CBP mount in alt/az.
        We assume the first file in the list has accurate initial mount pointings...
        """
        header_list = pft.open(test_file)
        keys = ['cbpmountalt', 'cbpmountaz', 'mountalpha', 'mountdelta']
        imps = {}
        logging.info('Inital pointings:')
        try:
            for k in keys:
                imps[k] = np.float(header_list[0].header[k])
                logging.info('%s: %6.2f' % (k, imps[k]))
        except KeyError:
            logging.info('No pointing headers found (this usually happens with old fits files). Setting imps to -99.')
            logging.info('If config.spots_can_move = True, the photometry code will do convolutions, which is slower.')
            for k in keys:
                imps[k] = -99
        header_list.close()
        return imps
       

    def _get_file_list(self):
        """
        List of fits files belonging to the scan.
        File_list MUST be sorted, because dark/light frames are 
        deterministic (e.g., 0 is dark, 1 is light, 2 is dark....)
        """
        file_list = glob.glob(os.path.join(self.config.data_location, '*.fits'))
        assert(len(file_list) > 0), 'No FITS files found in {}'.format(self.config.data_location)
        logging.info('Using fits files in: {}'.format(self.config.data_location))
        return sorted(file_list)

    def _get_dots(self):
        """
        Where are the initial spot locations?  
        Number is given by the shape of the <<basename>>_spots.txt file
        Locations are in the file itself
        """
        spot_file = os.path.join(self.config.data_location, self.base_name+'_spots.txt')
        assert(os.path.exists(spot_file)), 'Spot file {} not found!'.format(spot_file)
        return np.loadtxt(spot_file)


    def run_scan(self):
        """
        Process CBP exposures
        """
        # pool.map returns a list of "ExposureResults" objects
        # which can be used to load data into the attributes defined in 
        # the __init__ call
        with mp.Pool(processes=self.config.n_cpus) as pool:
            mapfunc = partial(process_exposure, self.config, self.pim)
            results = pool.map(mapfunc, self.image_numbers)

        exp_keys = ['laser_wavelength', 'file_name', 'exp_time', 'filter', 'charge', 'spectrum',
                    'n_spectra', 'spec_saturated']
        dot_keys = ['flux', 'raw_flux', 'flux_uncert', 'dot_loc']
        
        # Store results in either exposure-level attributes, or in attributes of the
        # Dot, if appropriate.
        for result in results:
            f = result.image_number
            # Because the image_numbers are every *other* file
            # We have to find the correct position in the output array
            i = np.int((f-1)/2)
            for k in exp_keys:
                if not self.config.process_spectra and k in ['spectrum', 'n_spectra', 'spec_saturated']:
                    continue
                self.__dict__[k][i] = result.__dict__[k]
            for k in dot_keys:
                for n,d in enumerate(self.dots):
                    d.__dict__[k][i] = result.__dict__[k][n]

        if self.config.results_sorting_key is not None:
            self._sort_results(self.config.results_sorting_key, exp_keys, dot_keys)

        # Post-processing:
        # Find wavelength values from calibrated spectrograph:
        # If we didn't process the spectra, we can't estimate wavelengths from them
        # so we won't have measured wavelengths --> no interpolating spline either.
        if self.config.process_spectra:
            for i,spec in enumerate(self.spectrum):
                self.measured_wavelength[i],_ = cbph.get_output_wavelength(self.config, spec)
            
            self._fit_wavelength_interp_spline()

        # Generate interpolating spline to infer wavelength where spectrum is saturated
        # If process_spectra is False, cbp_transmission is evaluated at laser wavelengths
        self._get_cbp_transmission()
        self._calculate_dot_throughputs()
        self._calculate_uncertainty()

        return # For now... TODO: Make sure this is all we want to do in run

    def _fit_wavelength_interp_spline(self):
        sat = self.spec_saturated
        unique_wavelengths = np.array(list(set(self.laser_wavelength[~sat])))
        unique_wavelengths = np.sort(unique_wavelengths)
        avg_output = np.zeros_like(unique_wavelengths)
        # Find the average MEASURED wavelength emitted by the laser at each requested wavelength
        # for exposures whose spectrograph data DOES NOT contain saturated values
        for i,w in enumerate(unique_wavelengths):
            avg_output[i] = np.mean(self.measured_wavelength[~sat][self.laser_wavelength[~sat]==w])
        self.wavelength_interp_spline = UnivariateSpline(unique_wavelengths, avg_output,
                                                         s=0, k=self.config.wavelength_interp_spline_deg)
        self.wavelength_splinefit = self.wavelength_interp_spline(self.laser_wavelength)

    def _sort_results(self, sorting_key, exp_keys, dot_keys):
        """
        Sometimes the scan does not proceed in order of increasing wavelength
        Nevertheless, it is convienient to work with an output that is sorted
        by (laser) wavelength (default value; can be changed)
        Have to make sure we use the same sorting vector (s) on both the dots and
        the per-exposure results, lest we accidentally switch data (since laser_wavelength
        is not a unique quantity between exposures)
        I don't know if this would actually occur, but better safe than sorry.
        """
        s = np.argsort(self.__dict__[sorting_key])
        for key in exp_keys:
            self.__dict__[key] = self.__dict__[key][s]
        for dot in self.dots:
            for key in dot_keys:
                dot.__dict__[key] = dot.__dict__[key][s]

    def _get_cbp_transmission(self):
        """
        Gets the cbp transmission at each wavelength 
        """
        if self.config.process_spectra:
            wl_grid = self.wavelength_splinefit
        else:
            wl_grid = self.laser_wavelength
            logging.warn('process_spectra==False. Evaluating CBP transmission at laser_wavelength')
        self.cbp_transmission, self.cbp_uncert = cc.get_cbp_transmission(wl_grid)

    def _calculate_dot_throughputs(self):
        for dot in self.dots:
            dot.raw_throughput = dot.flux / self.charge / self.cbp_transmission

    def _calculate_uncertainty(self):
        """
        This needs to be done on a per-dot basis because each dot has a different aperture photometry uncert
        """
        for dot in self.dots:
            tu, au, pu, cu = cbph.get_throughput_uncert(dot.flux_uncert, self.charge_uncert, dot.flux,
                                               self.charge, self.cbp_transmission, self.cbp_uncert)
            dot.total_uncert = tu
            dot.apphot_uncert = au
            dot.pd_uncert = pu
            dot.cbpt_uncert = cu

    def save(self, filename):
        with open(filename, 'wb') as my_file:
            pickle.dump(self.__dict__, my_file)

    @classmethod
    def load(cls, filename):
        """
        Factory returning a copy of the class as written to disk in
        the file "filename".
        """
        with open(filename, 'rb') as my_file:
            return cls(None, pickle.load(my_file))

class Dot():
    def __init__(self, number, n_images):
        self.number = number
        self.flux = np.zeros(n_images).astype(np.float)         # Total aperture counts - Sky counts
        self.raw_flux = np.zeros_like(self.flux)                # Total aperture counts
        self.flux_uncert = np.zeros_like(self.flux)             # Aperture uncertainty
        self.dot_loc = np.zeros((n_images, 2)).astype(np.float) # Dot location on focal plane
        self.raw_throughput = np.zeros_like(self.flux)          # Flux / Charge / CBP Transmission [a.u.]
        self.total_uncert = np.zeros_like(self.flux)            # Total estimated uncertainty on raw_throughput
        self.apphot_uncert = np.zeros_like(self.flux)           # Aper phot uncertianty contrib to total_uncert
        self.pd_uncert = np.zeros_like(self.flux)               # Photodiode uncertainty contrib to total_uncert
        self.cbpt_uncert = np.zeros_like(self.flux)             # CBP calibration uncertainty contrib to total_uncert


