import numpy as np
import astropy.io.fits as pft
import glob
import matplotlib.pyplot as plt
import os
import pickle
import cbp_helpers as cbph
import time
import cbp_calib as cc
import multiprocessing as mp
from functools import partial


def getStandardParams():
    params = {}
    params['ap_phot_rad'] = 45
    params['sky_rad_in'] = params['ap_phot_rad'] + 5
    params['sky_rad_out'] = params['sky_rad_in'] + 5
    params['search_rad'] = 10
    params['can_move'] = False
    params['gain'] = 1.
    params['min_charge'] = 0.5E-7
    params['read_noise'] = 3.
    return params

def getDark(filename, master_bias=None):
    if master_bias is None:
        master_bias = cbph.MasterBias()

    dark = pft.open(filename)
    dark_bias = master_bias(dark[0])
    return dark[0].data.astype(np.float) - dark_bias


def processImage(file_list, params, dot_locs, image_num):
    #  File IO + header parsing
    f = file_list[image_num]
    i = image_num
    master_bias = cbph.MasterBias()

    filename = file_list[i]
    dark_filename = file_list[i-1]
    dark_data = getDark(dark_filename)

    d = pft.open(f)
    wavelength = np.float(d[0].header['laserwavelength'])
    filt = np.int(d[0].header['FILTER'])
    print(os.path.split(f)[-1], wavelength, '{:.0f} of {:.0f}'.format((i+1)/2, len(file_list)/2))
    expTime = np.float(d[0].header['EXPTIME'])
    bias = master_bias(d[0])
    data = d[0].data.astype(np.float) - bias - dark_data
    data = data * params['gain']

    phd = d['PHOTOCOUNT'].data['phd']
    phd_time = d['PHOTOCOUNT'].data['time']
    bkg_charge = cbph.estimate_charge_bkg(phd_time, phd, expTime)
    charge = np.max(phd) - bkg_charge

    #  ====================================================
    #  Process Spectra
    spectrum, _ = cbph.reduceSpectra(d['SPECTRA'].data)

    #  =====================================================
    #  Aperture photometry
    phot_table, uncert = cbph.doAperturePhotometry(dot_locs,data,f,params)

    #  =====================================================
    #  Update info dictionary with new photometry + locations
    flux = phot_table['residual_aperture_sum']
    raw_flux = phot_table['aperture_sum']
    aper_uncert = uncert
    #  ====================================================
    #  PLOTTING
    cbph.makeDiagnosticPlots(data, dot_locs, params, f, wavelength, dark_data)
    cbph.makeDotHistograms(data, dot_locs, params['ap_phot_rad'], f, wavelength)
    cbph.makeDotImages(data, dot_locs, params['ap_phot_rad'], f, wavelength)
    cbph.makeSpectrumPlot(wavelength, spectrum[:,0], spectrum[:,1], f)

    return i, f, wavelength, filt, expTime, charge, spectrum, flux, raw_flux, aper_uncert


def processCBP(params=None, fits_file_path=None, make_plots=True, suffix='', show_final=True, no_ap_phot=False):
    """
    Routine to extract the throughput values from a CBP scan.  Output is stored in a dictionary, which is saved to the same location as the input files.

    :param params: Input dictionary containing extraction parameters.  If none is provided, use a default guess
    :param fits_file_path: Path to folder containing fits files.  Should not have a trailing /.
    :param make_plots: If True, make and save output plots, potentially erasing previous runs plots.
    :param suffix: Suffix to use for output files.  Can be used to differentiate output products from different processing runs of the same data
    :param show_final: If true, display the final plot of throughput vs wavelength.  Disabling this is useful for completely headless processing.
    :param no_ap_phot: Do throughput on a pixel-by-pixel bases.  Most useful with large pinhole/flatfields.
    """

    start_time = time.time()


    #  important parameters -- If not given, make assumptions
    if params is None:
        params = getStandardParams()

    #  Inputs
    if fits_file_path is None:
        raise ValueError('No fits_file_path provided')

    #  File location and checking
    root_name = os.path.split(fits_file_path)[-1]
    file_list = glob.glob(os.path.join(fits_file_path, '*.fits'))
    assert(len(file_list) > 0), 'No fits files found in {}'.format(fits_file_path)

    # input file list has to be sorted -- first file is dark, second is light etc
    file_list = sorted(file_list)

    # Since every other image is a dark, need to slice to get number of useful images
    n_images = len(file_list[1::2])

    # (manually) defines the location of the spots
    spot_file = os.path.join(fits_file_path, root_name+'_spots.txt')
    assert(os.path.exists(spot_file)), 'Spot file {} not found!'.format(spot_file)
    dot_locs = np.loadtxt(spot_file)

    #  Outputs
    #  output files + plots
    pkl_filename = os.path.join(fits_file_path, root_name + suffix + '.pkl')
    tpt_plot_name = os.path.join(fits_file_path, root_name + suffix + '.png')

    # initialize input dictionary
    info_dict = {}
    info_dict['dot_locs'] = dot_locs
    info_dict['params'] = params
    info_dict['run_name'] = os.path.split(fits_file_path)[-1]

    #  set up our output dictionary
    # different in the case of pix-by-pix and aperture modes
    if no_ap_phot:
        info_dict['pixel_data'] = {}
    else:
        for i, dot in enumerate(info_dict['dot_locs']):
            info_dict['dot%d' % i] = {}
            info_dict['dot%d' % i]['flux'] = np.zeros(n_images).astype(np.float)
            info_dict['dot%d' % i]['raw_flux'] = np.zeros(n_images).astype(np.float)
            info_dict['dot%d' % i]['aper_uncert'] = np.zeros(n_images).astype(np.float)
    info_dict['filename'] = np.zeros(n_images).astype(np.str)
    info_dict['dark_filename'] = np.zeros(n_images).astype(np.str)
    info_dict['filter'] = np.zeros(n_images).astype(np.int)
    info_dict['wavelengths'] = np.zeros(n_images).astype(np.float)
    info_dict['exp_times'] = np.zeros(n_images).astype(np.float)
    info_dict['charge'] = np.zeros(n_images).astype(np.float)
    info_dict['spectrum'] = np.zeros((n_images,994,2)).astype(np.float)

    #  slicing selects light images only
    #  then, flist[fnum-1] is the dark for flist[fnum]
    fnum = np.arange(0,len(file_list),1)[1::2]
    
    if no_ap_phot:
        pass
    else:
        with mp.Pool() as pool:
            mapfunc = partial(processImage, file_list, params, dot_locs)
            res = pool.map(mapfunc, fnum)
        for fnum, filename, wavelength, filt, expTime, charge, spectrum, flux, raw_flux, aper_uncert in res:
            i = np.int((fnum-1)/2) # image number for a given file number
            info_dict['filename'][i] = filename
            info_dict['wavelengths'][i] = wavelength
            info_dict['filter'][i] = filt
            info_dict['exp_times'][i] = expTime
            info_dict['charge'][i] = charge
            info_dict['spectrum'][i, :, :] = spectrum
            for s in range(len(dot_locs)):
                info_dict['dot%d' % s]['flux'][i] = flux[s]
                info_dict['dot%d' % s]['raw_flux'][i] = raw_flux[s]
                info_dict['dot%d' % s]['aper_uncert'][i] = aper_uncert[s]

    #  begin post-processing of photometry
    #  ========================================================

    # append nominal CBP system efficiency values to dict
    info_dict['cbp_transmission']  = cc.get_cbp_transmission(info_dict['wavelengths'])

    #  calculate throughputs
    charge_mask = info_dict['charge'] > params['min_charge']
    if not any(charge_mask):
        print('WARNING: NO CHARGE LEVELS EXCEED MIN_CHARGE {:.2e}'.format(params['min_charge']))
        charge_mask = np.ones_like(info_dict['charge']).astype(np.bool)
    #  ==============================================

    if make_plots:
        plt.figure(figsize=(24, 12))
    wavelength = info_dict['wavelengths']
    for i in range(len(info_dict['dot_locs'])):
        info_dict['charge_uncert'] = info_dict['charge']*0.01 + 50. * 1e-11
        x = info_dict['dot%d' % i]['flux'] / info_dict['charge']
        info_dict['dot%d' % i]['raw_tpt'] = np.asarray(x, dtype=np.float)
        info_dict['dot%d' % i]['rel_tpt'] = x/np.max(x[charge_mask])
        yerr = cbph.getTptUncert(info_dict['dot%d'%i]['aper_uncert'], info_dict['charge_uncert'],
                                 info_dict['dot%d'%i]['flux'], info_dict['charge'])
        info_dict['dot%d' % i]['rawtpt_uncert'] = yerr
        if make_plots:
            yerr = np.asarray(yerr)
            r = x / info_dict['cbp_transmission']
            r = r / np.max(r)
            plt.errorbar(wavelength, r ,yerr=yerr/np.max(x[charge_mask]),marker='o',
                         label='%d'%i,ls='',capsize=3,ms=3)

    with open(pkl_filename, 'wb') as myfile:
        pickle.dump(info_dict, myfile)

    #  make median throughput array + file:
    n = len(info_dict['dot_locs'])
    tptarr = np.zeros((len(wavelength), n)).astype(np.float)
    for i in range(len(info_dict['dot_locs'])):
        tptarr[:, i] = np.array(info_dict['dot%d'%i]['raw_tpt']) / np.max(info_dict['dot%d'%i]['raw_tpt'][charge_mask])
    tpts = np.median(tptarr, axis=1)
    asc_file_name = os.path.join(fits_file_path,root_name+suffix+'_median_tpt.txt')
    cbph.makeAsciiFile(wavelength, tpts, info_dict['charge'], fname=asc_file_name)

    print('Processing took {:6.1f} seconds.'.format(time.time()-start_time))


    if make_plots:
        plt.ylim(0,1.1)
        plt.axhline(y=0,ls='--')
        plt.legend(ncol=2)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Relative Throughput\nAdjusted for CBP Trans')
        plt.savefig(tpt_plot_name)
        if show_final:
            plt.show()
        else:
            plt.clf()
            plt.close()

        return info_dict
