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
params['use_flat'] = True
params['ronchi'] = False
params['can_move'] = True
params['gain'] = 3.
params['use_overscan'] = True

phot_rads = [45]#[20,25,30,35,40,45,50,55,60]
for pr in phot_rads:
    params = {}
    params['ap_phot_rad'] = pr
    params['sky_rad_in'] = params['ap_phot_rad'] + 5
    params['sky_rad_out'] = params['sky_rad_in'] + 5
    params['search_rad'] = 10
    params['use_flat'] = False
    params['ronchi'] = True
    params['can_move'] = True
    params['gain'] = 3.
    params['use_overscan'] = True
    info_dict = cap.processCBP(fits_file_path='C:\\Users\\Nick\\Documents\\CTIO_CBP\\20171006_RONCHI400_clear',
                   params=params,bkg_method='2d',flat_name='whiteflat.fits',suffix='_%d'%pr,make_plots=True)