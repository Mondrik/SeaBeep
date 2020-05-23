# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:09:59 2017

@author: Nick
"""

import cbp_apphot as cap
import pickle


params = {}
params['ap_phot_rad'] = 45
params['sky_rad_in'] = params['ap_phot_rad'] + 5
params['sky_rad_out'] = params['sky_rad_in'] + 5
params['search_rad'] = 10
params['can_move'] = True
params['gain'] = 1.
params['min_charge'] = 0.5e-7
params['read_noise'] = 16.
params['subtract_dark'] = True
params['pix2wave_fit_file'] = '/home/mondrik/python_modules/SeaBeep/stardice_analysis/data/pix2wave_chebfit.txt'
params['wavelength_adjust_file'] = '/home/mondrik/python_modules/SeaBeep/stardice_analysis/data/wavelength_adjust_function_cheb.txt'

if True:
    with open('../data/derivs.pkl', 'rb') as myfile:
        derivs = pickle.load(myfile)
        params['mount_derivatives'] = derivs
params['process_spectra'] = True

#phot_rads = [20, 25, 30, 35, 40, 50, 55, 60, 65]
phot_rads = [30]
#phot_rads = [45]


for pr in phot_rads:
    params['ap_phot_rad'] = pr
    params['sky_rad_in'] = 50
    params['sky_rad_out'] = params['sky_rad_in'] + 10
    params['search_rad'] = 30
    params['gain'] = 1.
    params['min_charge'] = 0.5e-7
    params['read_noise'] = 16.
    params['can_move'] = True
    info_dict = cap.processCBP(fits_file_path='/home/mondrik/CBP/paris_data/combination_scan',
                   params=params, suffix='_%d_incRN'%pr, make_plots=True, show_final=True)
