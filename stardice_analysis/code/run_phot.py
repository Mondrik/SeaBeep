# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:09:59 2017

@author: Nick
"""

import cbp_apphot as cap


params = {}
params['ap_phot_rad'] = 45
params['sky_rad_in'] = params['ap_phot_rad'] + 5
params['sky_rad_out'] = params['sky_rad_in'] + 5
params['search_rad'] = 10
params['can_move'] = True
params['gain'] = 1.
params['min_charge'] = 0.5e-7
params['read_noise'] = 16.

#phot_rads = [20,25,30,35,40,45,50]
phot_rads = [10, 55, 60, 65]
for pr in phot_rads:
    params = {}
    params['ap_phot_rad'] = pr
    params['sky_rad_in'] = params['ap_phot_rad'] + 5
    params['sky_rad_out'] = params['sky_rad_in'] + 5
    params['search_rad'] = 10
    params['can_move'] = False
    params['gain'] = 1.
    params['min_charge'] = 0.5e-7
    params['read_noise'] = 16.
    info_dict = cap.processCBP(fits_file_path='/home/mondrik/CBP/paris_data/combination_scan',
                   params=params, suffix='_%d'%pr, make_plots=True, show_final=False)
