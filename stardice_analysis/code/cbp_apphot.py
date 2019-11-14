import numpy as np
import astropy.io.fits as pft
import glob
import matplotlib.pyplot as plt
import os
import pickle
import cbp_helpers as cbph
import time
import cbp_calib as cc


def getStandardParams():
    params = {}
    params['ap_phot_rad'] = 45
    params['sky_rad_in'] = params['ap_phot_rad'] + 5
    params['sky_rad_out'] = params['sky_rad_in'] + 5
    params['search_rad'] = 10
    params['can_move'] = False
    params['gain'] = 1.
    params['min_charge'] = 0.5E-7
    return params


def processCBP(params=None, fits_file_path=None, make_plots=True, suffix=''):
    start_time = time.time()
    master_bias = cbph.MasterBias()
    #  important parameters -- If not given, make assumptions
    if params is None:
        params = getStandardParams()

    #  Inputs
    if fits_file_path is None:
        raise

    #  File location and checking
    root_name = os.path.split(fits_file_path)[-1]
    file_list = glob.glob(os.path.join(fits_file_path, '*.fits'))
    assert(len(file_list) > 0), 'No fits files found in %s' % fits_file_path
    file_list = sorted(file_list)    # input file list has to be sorted -- first file is dark, second is light etc

    spot_file = os.path.join(fits_file_path, root_name+'_spots.txt')
    assert(os.path.exists(spot_file)), 'Spot file %s not found!' % spot_file
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
    for i, dot in enumerate(info_dict['dot_locs']):
        info_dict['dot%d' % i] = {}
        info_dict['dot%d' % i]['flux'] = []
        info_dict['dot%d' % i]['raw_flux'] = []
        info_dict['dot%d' % i]['dot_loc'] = []
        info_dict['dot%d' % i]['aper_uncert'] = []
    info_dict['filename'] = []
    info_dict['dark_filename'] = []
    info_dict['wavelengths'] = []
    info_dict['exp_times'] = []
    info_dict['charge'] = []

    #  slicing selects light images only
    #  then, fnum-1 is the dark for that exposure
    for fnum, f in zip(np.arange(0, len(file_list), 1)[1::2], file_list[1::2]):
        #  ====================================================
        #  File IO + header parsing
        info_dict['filename'].append(f)
        info_dict['dark_filename'].append(file_list[fnum-1])
        dark = pft.open(file_list[fnum-1]) # should turn this into a "getDark" routine...
        dark_data = dark[0].data.astype(np.float)
        dark_bias = master_bias(dark[0])
        dark_data = dark_data - dark_bias
        d = pft.open(f)
        wavelength = np.float(d[0].header['laserwavelength'])
        print(os.path.split(f)[-1], wavelength, '{:.0f} of {:.0f}'.format((fnum+1)/2, len(file_list)/2))
        expTime = np.float(d[0].header['EXPTIME'])
        bias = master_bias(d[0])
        data = d[0].data.astype(np.float) - bias - dark_data
        data = data * params['gain']

        info_dict['wavelengths'].append(np.float(wavelength))
        info_dict['exp_times'].append(np.float(expTime))
        phd = d['PHOTOCOUNT'].data['phd']
        phd_time = d['PHOTOCOUNT'].data['time']
        bkg_charge = cbph.estimate_charge_bkg(phd_time, phd, expTime)
        # print('Initial phd charge: {} bkg charge: {}'.format(np.max(phd), bkg_charge))
        info_dict['charge'].append(np.max(phd)-bkg_charge)
        #  ====================================================

        #  ====================================================
        #  determine spot locations
        # new_locs = cbph.getNewLocs(data, info_dict, params)
        new_locs = dot_locs
        #  =====================================================

        #  =====================================================
        #  Aperture photometry
        phot_table, error = cbph.doAperturePhotometry(new_locs,data,f,params)
        #  =====================================================

        #  Update info dictionary with new photometry + locations
        #  =====================================================
        for i in range(len(info_dict['dot_locs'])):
            info_dict['dot%d' % i]['dot_loc'].append(new_locs[i])
            info_dict['dot%d' % i]['flux'].append(phot_table['residual_aperture_sum'][i])
            info_dict['dot%d' % i]['raw_flux'].append(phot_table['aperture_sum'][i])
            info_dict['dot%d' % i]['aper_uncert'].append(error[i])
        info_dict['dot_locs'] = new_locs
        #  ====================================================

        #  ====================================================
        #  PLOTTING
        cbph.makeDiagnosticPlots(data, new_locs, params, f, wavelength, dark_data)
        cbph.makeDotHistograms(data, new_locs, params['ap_phot_rad'], f, wavelength)
        cbph.makeDotImages(data, new_locs, params['ap_phot_rad'], f, wavelength)
        # cbph.makeSpectrumPlot(xmlDict,wavelength,f)
        #  ====================================================

    #  begin post-processing of photometry
    #  ========================================================
    #  sort output quantities by wavelength
    info_dict['wavelengths'] = np.asarray(info_dict['wavelengths'])
    info_dict['exp_times'] = np.asarray(info_dict['exp_times'])
    info_dict['charge'] = np.asarray(info_dict['charge'])

    # append nominal CBP system efficiency values to dict
    info_dict['cbp_transmission']  = cc.get_cbp_transmission(info_dict['wavelengths'])

    #  calculate throughputs
    for i,dot in enumerate(info_dict['dot_locs']):
        info_dict['dot%d' % i]['flux'] = np.asarray(info_dict['dot%d' % i]['flux'],dtype=np.float)
        info_dict['dot%d' % i]['raw_flux'] = np.array(info_dict['dot%d' % i]['raw_flux'],dtype=np.float)
        info_dict['dot%d' % i]['dot_loc'] = np.asarray(info_dict['dot%d' % i]['dot_loc'])
        info_dict['dot%d' % i]['aper_uncert'] = np.asarray(info_dict['dot%d' % i]['aper_uncert'])
    info_dict['filename'] = np.asarray(info_dict['filename'],dtype=np.str)
    charge_mask = info_dict['charge'] > params['min_charge']
    #  ==============================================

    if make_plots:
        plt.figure(figsize=(12, 12))
    wavelength = info_dict['wavelengths']
    for i in range(len(info_dict['dot_locs'])):
        info_dict['charge_uncert'] = info_dict['charge']*0.01 + 50. * 1e-11
        x = info_dict['dot%d' % i]['flux'] / info_dict['charge']
        info_dict['dot%d' % i]['raw_tpt'] = np.asarray(x, dtype=np.float)
        info_dict['dot%d' % i]['rel_tpt'] = x/np.max(x[charge_mask])
#        yerr = info_dict['dot%d'%i]['aper_uncert'] / info_dict['charge'] / np.max(x[charge_mask])
        yerr = cbph.getTptUncert(info_dict['dot%d'%i]['aper_uncert'], info_dict['charge_uncert'],
                                 info_dict['dot%d'%i]['flux'], info_dict['charge'])
        info_dict['dot%d' % i]['rawtpt_uncert'] = yerr
        if make_plots:
            yerr = np.asarray(yerr)
            r = x / info_dict['cbp_transmission']
            r = r / np.max(r)
            plt.errorbar(wavelength, r ,yerr=yerr/np.max(x[charge_mask]),marker='o',
                         label='%d'%i,ls='-',capsize=3,ms=3)

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

    if make_plots:
        plt.ylim(0,1.1)
        plt.axhline(y=0,ls='--')
        plt.legend(ncol=2)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Relative Throughput\nAdjusted for CBP Trans')
        plt.savefig(tpt_plot_name)
        plt.show()

    print('Processing took {:6.1f} seconds.'.format(time.time()-start_time))
    return info_dict
