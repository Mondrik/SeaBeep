"""
Collection of functions (and a class) generally needed to reduce CBP data.
"""

import numpy as np
import photutils as pt
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import os
import astropy.io.fits as pft
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.stats import sigma_clipped_stats
from astropy.modeling import fitting
from astropy.modeling.models import Gaussian1D, Polynomial1D
import logging


# Should probably think about removing some of these options or adding them as config parameters
def find_center(data, guess, search_radius=20, kernel=None, sigma_x=10, sigma_y=10, enforce_colinear=False):
    """
    Given a set of initial guesses for the locations of spots, convolve the image with a kernel and search the result for a peak within a specified radius

    :param data: image to search
    :param guess: [row, column] guess for the location of the spot in pixel space. Must be an numpy array of ints.
    :param search_radius: number of pixels away from guess to search.  Symmetric in x and y.
    :param kernel: If not None, use provided kernel to convolve data.  Must be useable with astropy convolution function
    :param sigma_x: If kernel is None, use this to set sigma_x for the 2D gaussian smoothing kernel
    :param sigma_y: If kernel is None, use this to set sigma_y for the 2D gaussian smoothing kernel
    :param enforce_colinear: Bool. If True, only return new locations if *all* shift vectors are (approx) colinear and of equal length.

    :return locs: [row, column] location of the maximum for the convolved image within the region defined by guess and search_radius
    """
    if kernel is None:
        kernel = Gaussian2DKernel(x_stddev=sigma_x, y_stddev=sigma_y)

    guess = guess.astype(np.int)
    locs = np.zeros_like(guess)
    conv_image = convolve(data, kernel)
    for i,g in enumerate(guess):
        region = conv_image[g[0]-search_radius:g[0]+search_radius+1, g[1]-search_radius:g[1]+search_radius+1]
        index = np.array(np.unravel_index(np.argmax(region, axis=None), region.shape)).astype(np.int)
        loc = index + g - [search_radius, search_radius]
        locs[i] = loc

    if enforce_colinear:
        # now, if we only want to move when things are colinear, need to define *how* colinear all
        # points are on avg:
        shift_vectors = locs - guess
        dot_products = []
        for i in range(shift_vectors.shape[0]-1):
            for j in range(i+1, shift_vectors.shape[0]):
                dot_products.append(np.sum(shift_vectors[i,:] * shift_vectors[j,:]))
        dot_products = np.array(dot_products)
        mean_shift = np.mean(dot_products)
        std_shift = np.std(dot_products)
        if mean_shift / std_shift < 5.:
            return guess
        else:
            return guess + np.mean(shift_vectors, axis=0)
        
    return locs


class MasterBias():
    """
    Class that contains code to generate a bias frame from a given StarDICE exposure
    """
    def __init__(self, filename='../data/master_bias.fits'):
        self.header_key = 'IMREDMB'
        self.filename = filename
        with pft.open(filename) as fid:
            self.data = fid[0].data
            self.header = fid[0].header
        self.m = self.header['MEANPOWR']
        self.slope = self.header['POWRSLOP']

    def __call__(self, image, force=False):
        if 'REG_POWR' in image.header:
            return self.data + self.slope * (image.REG_POWR - self.m)
        elif 'REG-POWR' in image.header:
            return self.data + self.slope * (image.header['REG-POWR'] - self.m)
        else:
            logging.error('Peletier Power not registered in image header. Bias level cannot be computed accurately.')
            if force:
                return self.data
            else:
                raise KeyError('REG_POWR missing in image header')

def do_aperture_photometry(config, data, locs):
    aplocs = []
    for i, pair in enumerate(locs):
        aplocs.append(pair[::-1])

    apertures = pt.CircularAperture(aplocs, r=config.ap_phot_rad)

    # Estimate sky background as the (sigma clipped) median value of the pixels in the sky annulus
    sky_apertures = pt.CircularAnnulus(aplocs, r_in=config.sky_rad_in, r_out=config.sky_rad_out)
    sky_masks = sky_apertures.to_mask(method='center')
    sky_per_pix = np.zeros(len(sky_masks))
    for i,mask in enumerate(sky_masks):
        sky_data = mask.multiply(data)
        if sky_data is None:
            sky_per_pix[i] = np.nan
            if config.cbp_moves:
                logging.warn('sky_data is None, which probably means one of the spots has wandered off of the detector.  If this is expected, you\'re probably okay to continue.')
            else:
                logging.error('sky_data is None. The CBP is not allowed to move, so this should never happen.  Is one of the default spot locations off of the detector?')
            continue
        sky_data_1d = sky_data[mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(sky_data_1d)
        sky_per_pix[i] = median_sigclip
    phot_table = pt.aperture_photometry(data, apertures, method='subpixel', subpixels=5)
    phot_table['sky_bkg'] = sky_per_pix * apertures.area

    phot_table['residual_aperture_sum'] = phot_table['aperture_sum'] - phot_table['sky_bkg']

    # Noise calculation
    # Noise = Sqrt(Signal + npix*(1 + npix/nbkg)*(sky_per_pix + read_noise**2))
    # Read noise**2 is multiplied by 2, since we are doing photometry on a diff image
    npix = np.ceil(apertures.area)
    nbkg = np.ceil(sky_apertures.area)
    uncert = np.sqrt(phot_table['residual_aperture_sum'] + npix*(1 + npix/nbkg)*(sky_per_pix + 2.*config.read_noise**2.))
    ras = phot_table['residual_aperture_sum']
    with np.printoptions(precision=3, suppress=True):
        logging.debug(f'NPIX: {npix:.0f} NBKG_PIX: {nbkg:.0f}\nSKY PER PIX: {sky_per_pix}\n{ras}')
    return phot_table, uncert


def get_throughput_uncert(aper_uncert,charge_uncert,flux,charge,cbpt,cbpt_uncert):
    # From error propagation, uncert is of the form:
    # sigma Transmission = SQRT( (sigma_CCD/Q_CCD)^2 + (Q_CCD*sigma_PD/Q_PD^2)^2 )
    flux_uncert = np.sqrt((aper_uncert/charge/cbpt)**2.)
    pd_uncert = np.sqrt((flux*charge_uncert/charge**2./cbpt)**2.)
    cbpt_uncert = np.sqrt((flux*cbpt_uncert/charge/cbpt**2.)**2.)
    uncert = np.sqrt( flux_uncert**2. + pd_uncert**2. + cbpt_uncert**2. )
    return uncert, flux_uncert, pd_uncert, cbpt_uncert


def reduce_spectra(specs):
    """
    Stacks all spectra from to provided np record array to form a co-added spectrum
    This record array has one key labeled "wl" (wavelength), and an arbitrary number
    of keys like "fluxXX" where XX is some integer.
    If any are saturated (=65535), we set the saturation flag to true, so we know to
    exclude these from fitting later.
    """
    wavelengths = specs['wl']
    counts = np.zeros_like(wavelengths)
    n_specs = 0
    has_saturated_values = False
    while True:
        try:
            if any(specs['flux%d'%n_specs] == 65535):
                has_saturated_values = True
            counts += specs['flux%d'%n_specs]
            n_specs += 1
        except:
            break
    collapsed_spec = np.column_stack((wavelengths, counts))

    return collapsed_spec, n_specs, has_saturated_values

def get_output_wavelength(config, spectrum):

    pix2wave_fit = np.loadtxt(config.pix2wave_fit_file)
    pix2wave = np.polynomial.chebyshev.Chebyshev(pix2wave_fit)
    
    pix = np.arange(0, len(spectrum[:,1]), 1)
    # Starting pixel is the peak of the co-added spectrum
    center_pix = np.argmax(spectrum[:,1])

    mask_condition = np.abs(pix - pix[center_pix]) > config.spec_fit_region_size
    pix_masked = np.ma.masked_where(mask_condition, pix)
    counts_masked = np.ma.masked_where(mask_condition, spectrum[:,1])
    mask_center = np.argmax(counts_masked)

    fitter = fitting.LevMarLSQFitter()
    offset = Polynomial1D(degree=0)
    g = Gaussian1D(mean=pix_masked[center_pix], amplitude=counts_masked.max(), stddev=4.)
    model = g + offset
    bf_model = fitter(model, pix, spectrum[:,1])
    model_center_pix = bf_model.mean_0
    return pix2wave(model_center_pix), bf_model

def estimate_charge_bkg(time, charge, exptime, nprepost=10):
    fit = np.polyfit(time[:nprepost], charge[:nprepost], deg=1)
    bkg_charge = charge[0] + exptime*fit[0]
    # x = np.linspace(np.min(time), np.max(time), 1000)
    # p = np.poly1d(fit)
    # plt.plot(time, charge, 'ro')
    # plt.plot(x, p(x), '-k')
    # plt.show()
    return bkg_charge
