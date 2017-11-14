# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 23:06:38 2017

@author: Nick
"""

import numpy as np
import astropy.io.fits as pft
import photutils as pt
import glob
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import george
from scipy.stats import mode
import matplotlib.patches as patch
    
def findCenter(data,guess,size=5,search_size=50):
    lower = guess[0]-search_size
    bri = -99
    while lower < guess[0]+search_size:
        left = guess[1] - search_size
        while left < guess[1]+search_size:
            tot = np.sum(data[lower:lower+size,left:left+size].flatten())
            if tot > bri:
                bri = tot
                colcent = left + np.int(size/2)
                rowcent = lower + np.int(size/2)
            left += 1
        lower += 1
    return rowcent,colcent
    
def do_aperture_photometry(locs,data,rad=15,sky_rad_in=45,sky_rad_out=65):
    error = np.sqrt(data)
    
    aplocs = []
    for i,pair in enumerate(locs):
        aplocs.append(pair[::-1])

    apertures = pt.CircularAperture(aplocs,r=rad)
    
    bkg_arr = data.copy()
    bkg = []
    for loc in locs:
        bkg_arr[loc[0]-sky_rad_in:loc[0]+sky_rad_in,loc[1]-sky_rad_in:loc[1]+sky_rad_in] = np.nan
#        bkg.append(mode(bkg_arr[loc[0]-sky_rad_out:loc[0]+sky_rad_out,loc[1]-sky_rad_out:loc[1]+sky_rad_out],
#                        nan_policy='omit',axis=None)[0][0])
        bkg.append(np.nanmedian(bkg_arr[loc[0]-sky_rad_out:loc[0]+sky_rad_out,loc[1]-sky_rad_out:loc[1]+sky_rad_out].flatten()))
#    plt.imshow(bkg_arr,origin='lower',vmax=3000)
#    plt.colorbar()
    #for loc in locs:
    #    plt.gca().add_patch(patch.Rectangle((loc[1]-sky_rad_out,loc[0]-sky_rad_out),2*sky_rad_out,2*sky_rad_out,color='k',alpha=0.5))
#    plt.show()
    bkg = np.array(bkg)
    
    usemean = False
    if usemean:
        bkg_annulus = pt.CircularAnnulus(aplocs,r_in=sky_rad_in,r_out=sky_rad_out)
        apers = [apertures,bkg_annulus]
    else:
        apers = apertures
    
    phot_table = pt.aperture_photometry(data,apers,method='subpixel',subpixels=10,error=error)
    if usemean:
        bkg_mean = phot_table['aperture_sum_1'] / bkg_annulus.area()
        bkg_sum = bkg_mean * apertures.area()
    else:
        bkg_sum = bkg * apertures.area()
#    print('backgrounds:', bkg,np.array(bkg_mean))

    if usemean:
        final_sum = phot_table['aperture_sum_0'] - bkg_sum
    else:
        final_sum = phot_table['aperture_sum'] - bkg_sum
    phot_table['residual_aperture_sum'] = final_sum
#    print(phot_table['residual_aperture_sum'])
#    print(phot_table.keys())
    #area_ratio = bkg_annulus.area()/apertures.area()
    area_ratio = (sky_rad_out**2.-sky_rad_in**2.)/apertures.area()
    if usemean:
        error_final = np.sqrt(phot_table['aperture_sum_err_0']**2.+area_ratio*np.sqrt(bkg)**2.)
    else:
        error_final = phot_table['aperture_sum_err']
    return phot_table,bkg,error_final

def make_aux_plots(wavelength,combine,info_dict):
    plt.figure()
    plt.title('FLUX')
    plt.plot(wavelength,combine/np.max(combine),'-ko',label='Keith')
    for i in range(len(info_dict['dot_locs'])):
        plt.plot(wavelength,info_dict['dot%d' % i]['flux']/np.max(info_dict['dot%d' % i]['flux']),label='flux%d' % i)
    plt.ylim(0,1.01)
    plt.ylabel('Rel. Value.')
    plt.xlabel('Wavelength')
    plt.legend()
    plt.show(block=False)
    
    plt.figure()
    plt.title('EXP TIME')
    plt.plot(wavelength,combine/np.max(combine),'-ko',label='Keith')
    plt.plot(wavelength,info_dict['exp_times']/np.max(info_dict['exp_times']),'-ro',label='Exptime')
    plt.ylim(0,1.01)
    plt.xlabel('wavelength')
    plt.ylabel('Rel. Value')
    plt.legend()
    plt.show(block=False)
    
    plt.figure()
    plt.title('BACKGROUND')
    plt.plot(wavelength,combine/np.max(combine),'-ko',label='Keith')
    plt.plot(wavelength,info_dict['bkg']/np.mean(info_dict['bkg']),'-ro',
             label='bkg %d' % np.rint(np.max(info_dict['bkg'])))
    plt.ylim(0,1.01)
    plt.xlabel('wavelength')
    plt.ylabel('Rel. Value')
    plt.legend()
    plt.show(block=False)
    
    plt.figure()
    plt.title('SKY')
    plt.plot(wavelength,combine/np.max(combine),'-ko',label='Keith')
    for i in range(len(info_dict['dot_locs'])):
        plt.plot(wavelength,info_dict['dot%d' % i]['bkg_mean']/np.max(info_dict['dot%d' % i]['bkg_mean']),
                 label='%d'%i)
    plt.ylim(0,1.01)
    plt.xlabel('wavelength')
    plt.ylabel('Rel. Value')
    plt.show()
    
def get_GP_model(info_dict):
    xobs = np.empty(0)
    yobs = np.empty(0)
    eobs = np.empty(0)
    
    for i in range(len(info_dict['dot_locs'][:-1])):
        xobs = np.concatenate((xobs,info_dict['wavelengths']))
        yobs = np.concatenate((yobs,info_dict['dot%d'%i]['rel_tpt']))
        eobs = np.concatenate((eobs,info_dict['dot%d'%i]['reltpt_uncert']))
        
    xobs = xobs[1:]
    yobs = yobs[1:]/np.max(yobs[1:])
    eobs = eobs[1:]
    
    np.savetxt('test.dat',np.column_stack((xobs,yobs,eobs)))

    k = 1.0 * george.kernels.ExpSquaredKernel(1.0)
    gp = george.GP(k)
    gp.compute(xobs,yerr=eobs)
    
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(yobs)
    
    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(yobs)
    
    result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    print(result)
    gp.set_parameter_vector(result.x)
    
    xplot = np.linspace(525,725,300)
    yplot,yerr = gp.predict(yobs,xplot,return_var=True,return_cov=False)
    plt.fill_between(xplot,yplot+2.*yerr,yplot-2.*yerr,color='k',alpha=0.2)
    plt.show()