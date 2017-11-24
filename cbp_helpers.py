# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 23:06:38 2017

@author: Nick
"""

import numpy as np
import photutils as pt
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import os
    
#search via a convolution for center
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

def getNewLocs(data,info_dict,params):
    if params['can_move']:
        if not params['ronchi']:
            #We allow all spots to move, but must move as a fixed grid -- therefore
            #only move by the median displacement vector from current positions
            new_locs = []
            for i,pair in enumerate(info_dict['dot_locs']):
                myrow,mycol = findCenter(data,pair,search_size=params['search_rad'])
                new_locs.append([myrow,mycol])  
            new_locs = np.array(new_locs)
            vec = new_locs - info_dict['dot_locs']
            vec = np.rint(np.median(vec,axis=0))
            new_locs = info_dict['dot_locs'] + vec
        else:
            new_locs = info_dict['dot_locs']
            myrow,mycol = findCenter(data,new_locs[0],search_size=params['search_rad'])
            new_locs[0] = [myrow,mycol]
            new_locs[1] = new_locs[0]
            #guesstimate of ronchi wavelength solution -- doesn't need to be 
            #perfect, just good enough
            new_locs[1][1] += np.float(info_dict['wavelengths'][-1])*0.53 - 17.61538
    else:
        if not params['ronchi']:
            new_locs = info_dict['dot_locs']
        else:
            new_locs = info_dict['dot_locs']
            new_locs[1] = new_locs[0]
            new_locs[1][1] += np.float(info_dict['wavelengths'][-1])*0.53 - 17.61538
    return new_locs

def getBackground(data,fitsfilename,method='2d',box_size=50,params=None,locs=None):
    if method == '2d':
        bkg_estimator = pt.MedianBackground()
        data_mask = (data<1.1*np.median(data))
        bkg_maskarr = np.ma.array(data,mask=~data_mask)
        bkg = pt.Background2D(bkg_maskarr, (box_size,box_size), filter_size=(3,3), sigma_clip=None,
                              bkg_estimator=bkg_estimator)
        bkg_median = bkg.background_median
        plt.imshow(bkg.background,origin='lower',vmin=bkg_median-100,
                   vmax=bkg_median+100,cmap='Greys')
        plt.colorbar(orientation='horizontal')
        if not os.path.exists(os.path.join(os.path.dirname(fitsfilename),'background')):
            os.makedirs(os.path.join(os.path.dirname(fitsfilename),'background'))
        savepath = os.path.join(os.path.join(os.path.dirname(fitsfilename),'background'),
                                'bkg_'+os.path.split(fitsfilename[:-5])[-1]+'.png')
        plt.savefig(savepath)
        plt.clf()
        plt.close()
    #aperture background method not currently working, don't use!!!!!
    elif method == 'aperture':
        bkg_arr = data.copy()
        bkg = []
        for loc in locs:
            bkg_arr[loc[0]-params['sky_rad_in']:loc[0]+params['sky_rad_in'],
                    loc[1]-params['sky_rad_in']:loc[1]+params['sky_rad_in']] = np.nan
            bkg.append(np.nanmedian(bkg_arr[loc[0]-params['sky_rad_out']:loc[0]+params['sky_rad_out'],
                                            loc[1]-params['sky_rad_out']:loc[1]+params['sky_rad_out']].flatten()))

        plt.imshow(bkg_arr,origin='lower',vmin=np.nanmedian(bkg_arr)-100,vmax=np.nanmedian(bkg_arr)+100)
        plt.colorbar()
        for loc in locs:
            plt.gca().add_patch(patch.Rectangle((loc[1]-params['sky_rad_out'],loc[0]-params['sky_rad_out']),
                                2*params['sky_rad_out'],2*params['sky_rad_out'],color='k',alpha=0.5))
        if not os.path.exists(os.path.join(os.path.dirname(fitsfilename),'background')):
            os.makedirs(os.path.join(os.path.dirname(fitsfilename),'background'))
        savepath = os.path.join(os.path.join(os.path.dirname(fitsfilename),'background'),
                                'bkg_'+os.path.split(fitsfilename[:-5])[-1]+'.png')
        plt.savefig(savepath)
        plt.clf()
        plt.close()
        bkg = np.array(bkg)
    else:
        raise ValueError('Method %s not supported.' % method)
        
    return bkg
        
def doAperturePhotometry(locs,data,fitsfilename,params,bkg_method='2d'):
    error = np.sqrt(data)
    aplocs = []
    for i,pair in enumerate(locs):
        aplocs.append(pair[::-1])

    apertures = pt.CircularAperture(aplocs,r=params['ap_phot_rad'])
    bkg = getBackground(data,fitsfilename,method=bkg_method,locs=locs,params=params)
    phot_table = pt.aperture_photometry(data-bkg.background,apertures,method='subpixel',subpixels=10,error=error)
    phot_table['residual_aperture_sum'] = phot_table['aperture_sum']
    error_final = phot_table['aperture_sum_err']
    return phot_table,bkg,error_final

def makeAuxPlots(wavelength,combine,info_dict):
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

def makeDiagnosticPlot(data,locs,params,fitsfilename,wavelength,savepath='./'):
    plt.figure(figsize=(12,12))
    plt.imshow(data,origin='lower',vmin=-200,vmax=500)
    wedges = []
    
    for j,loc in enumerate(locs):
        wedge = patch.Wedge(center=loc[::-1],theta1=0,theta2=360.,
                                r=params['sky_rad_out'],width=params['sky_rad_out']-params['sky_rad_in'],
                                color='r',alpha=0.5)
        wedges.append(wedge)
        plt.text(loc[1]+params['sky_rad_out']+10,loc[0]-params['sky_rad_out']-10,'%d' % j,
                 color='w',size=16)
    for wedge in wedges:
        plt.gca().add_patch(wedge)

    circs = []
    for loc in locs:
        circ = patch.Circle(xy=loc[::-1],
                                radius=params['ap_phot_rad'],color='k',alpha=0.5)
        circs.append(circ)
    for circ in circs:
        plt.gca().add_patch(circ)

    plt.colorbar(orientation='horizontal')
    plt.title(os.path.split(fitsfilename[:-5])[-1])
    plt.text(800,100,'%snm' % wavelength,color='w',size=16)
    if not os.path.exists(os.path.join(os.path.dirname(fitsfilename),'diagnostics')):
        os.makedirs(os.path.join(os.path.dirname(fitsfilename),'diagnostics'))
    savepath = os.path.join(os.path.join(os.path.dirname(fitsfilename),'diagnostics'),
                            os.path.split(fitsfilename[:-5])[-1]+'.png')
    plt.savefig(savepath)
    plt.clf()
    plt.close()