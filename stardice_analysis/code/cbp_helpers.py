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
from xml.etree import ElementTree as ET

#search via a convolution for center
def findCenter(data,guess,size=5,search_size=50):
    lower = np.int(guess[0]-search_size)
    bri = -99
    while lower < guess[0]+search_size:
        left = np.int(guess[1] - search_size)
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
        # We allow all spots to move, but must move as a fixed grid -- therefore
        # only move by the median displacement vector from current positions
        new_locs = []
        for i, pair in enumerate(info_dict['dot_locs']):
            myrow, mycol = findCenter(data, pair, search_size=params['search_rad'])
            new_locs.append([myrow,mycol])
        new_locs = np.array(new_locs)
        vec = new_locs - info_dict['dot_locs']
        vec = np.rint(np.median(vec, axis=0))
        new_locs = info_dict['dot_locs'] + vec
    else:
        new_locs = info_dict['dot_locs']
    return new_locs

def doAperturePhotometry(locs, data, fitsfilename, params):
    aplocs = []
    for i, pair in enumerate(locs):
        aplocs.append(pair[::-1])

    apertures = pt.CircularAperture(aplocs, r=params['ap_phot_rad'])
    uncert_table = pt.aperture_photometry(data, apertures, method='subpixel', subpixels=10)
    phot_table = pt.aperture_photometry(data,apertures,method='subpixel',subpixels=10)
    phot_table['residual_aperture_sum'] = phot_table['aperture_sum']

    uncert_final = np.sqrt(uncert_table['aperture_sum'])
    return phot_table, uncert_final


def getTptUncert(aper_uncert,charge_uncert,flux,charge,p=0.84):
    # p=0.84 bc 1-sigma (assuming normality) corresponds to 68% of the integral of the population
    # so (since tan is an odd fn) 1-sigma errorbars are analogous to calculating the 0 +/- 16/84 quantiles to
    # contain 68% of the random values from the PDF.
    # sigmax = aper_uncert
    # sigmay = charge_uncert
    # return sigmax/sigmay * np.tan(np.pi*(p-0.5)) #quantile function for Cauchy PDF
    return np.sqrt((flux/charge)**2. * ((aper_uncert/flux)**2. + (charge_uncert/charge)**2.))


def makeDiagnosticPlots(data, locs, params, fitsfilename, wavelength, savepath='./'):
    plt.figure(figsize=(12, 12))
    plt.imshow(data, origin='lower', vmin=-100, vmax=100)
    wedges = []

    for j, loc in enumerate(locs):
        wedge = patch.Wedge(center=loc[::-1],theta1=0, theta2=360.,
                            r=params['sky_rad_out'], width=params['sky_rad_out']-params['sky_rad_in'],
                            color='r', alpha=0.5)
        wedges.append(wedge)
        plt.text(loc[1]+params['sky_rad_out']+10, loc[0]-params['sky_rad_out']-10, '%d' % j,
                 color='w', size=16)
    for wedge in wedges:
        plt.gca().add_patch(wedge)

    circs = []
    for loc in locs:
        circ = patch.Circle(xy=loc[::-1],
                            radius=params['ap_phot_rad'], color='k', alpha=0.5)
        circs.append(circ)
    for circ in circs:
        plt.gca().add_patch(circ)

    plt.colorbar(orientation='horizontal')
    plt.title(os.path.split(fitsfilename[:-5])[-1])
    plt.text(700, 50, '%snm' % wavelength, color='w', size=16)
    if not os.path.exists(os.path.join(os.path.dirname(fitsfilename), 'diagnostics')):
        os.makedirs(os.path.join(os.path.dirname(fitsfilename), 'diagnostics'))
    if not os.path.exists(os.path.join(os.path.dirname(fitsfilename), 'diagnostics', 'images')):
        os.makedirs(os.path.join(os.path.dirname(fitsfilename), 'diagnostics', 'images'))
    savepath = os.path.join(os.path.join(os.path.dirname(fitsfilename), 'diagnostics', 'images'),
                            os.path.split(fitsfilename[:-5])[-1]+'.png')
    plt.savefig(savepath)
    plt.clf()
    plt.close()

    if not os.path.exists(os.path.join(os.path.dirname(fitsfilename), 'diagnostics', 'histograms')):
        os.makedirs(os.path.join(os.path.dirname(fitsfilename), 'diagnostics', 'histograms'))
    savepath = os.path.join(os.path.join(os.path.dirname(fitsfilename), 'diagnostics', 'histograms'),
                            os.path.split(fitsfilename[:-5])[-1]+'.png')
    plt.title('%snm' % wavelength)
    plt.hist(data.flatten(), bins=100, range=[-200, 1000], histtype='step', color='k')
    plt.axvline(np.median(data), ls='--', color='r')
    plt.savefig(savepath)
    plt.clf()
    plt.close()


def makeDotHistograms(data, locs, box_size, fitsfilename, wavelength, savepath='./'):
    locs = np.asarray(locs,dtype=np.int)
    plt.figure(figsize=(12,12))
    for i,loc in enumerate(locs):
        rmin = loc[0]-box_size
        rmax = loc[0]+box_size
        cmin = loc[1]-box_size
        cmax = loc[1]+box_size
        region = data[rmin:rmax,cmin:cmax].flatten()
        plt.hist(region,bins=20,range=[-200,65535],histtype='step',label='%d'%i)
    plt.legend(ncol=2)
    plt.title('%snm' % wavelength)
    if not os.path.exists(os.path.join(os.path.dirname(fitsfilename),'diagnostics')):
        os.makedirs(os.path.join(os.path.dirname(fitsfilename),'diagnostics'))
    if not os.path.exists(os.path.join(os.path.dirname(fitsfilename),'diagnostics','local_histograms')):
        os.makedirs(os.path.join(os.path.dirname(fitsfilename),'diagnostics','local_histograms'))
    savepath = os.path.join(os.path.join(os.path.dirname(fitsfilename),'diagnostics','local_histograms'),
                            os.path.split(fitsfilename[:-5])[-1]+'.png')
    plt.yscale('log')
    plt.savefig(savepath)
    plt.clf()
    plt.close()

def makeDotImages(data, locs, box_size, fitsfilename, wavelength, savepath='./'):
    locs = np.asarray(locs,dtype=np.int)
    ncolplts = 3
    nrowplts = np.ceil(len(locs)/ncolplts).astype(np.int)
    fig,ax = plt.subplots(nrowplts,ncolplts)
    fig.set_size_inches((16,8))
    ax1 = fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.title('%snm'%wavelength)
    for i,loc in enumerate(locs):
        row = np.floor(np.float(i)/np.float(ncolplts)).astype(np.int)
        col = i % ncolplts
        region = data[loc[0]-box_size:loc[0]+box_size,loc[1]-box_size:loc[1]+box_size]
        img = ax[row,col].imshow(region,vmin=-200,vmax=200,origin='lower')
        ax[row,col].text(5,5,'%d' % i,
                 color='w',size=16)
    while col < ncolplts:
        ax[row,col].axis('off')
        col += 1
    fig.colorbar(img, ax=ax.ravel().tolist())
    if not os.path.exists(os.path.join(os.path.dirname(fitsfilename),'diagnostics')):
        os.makedirs(os.path.join(os.path.dirname(fitsfilename),'diagnostics'))
    if not os.path.exists(os.path.join(os.path.dirname(fitsfilename),'diagnostics','local_images')):
        os.makedirs(os.path.join(os.path.dirname(fitsfilename),'diagnostics','local_images'))
    savepath = os.path.join(os.path.join(os.path.dirname(fitsfilename),'diagnostics','local_images'),
                            os.path.split(fitsfilename[:-5])[-1]+'.png')
    plt.savefig(savepath)
    plt.clf()
    plt.close()
#
def filterCosmics(data):
    i  = 0
    count = 1000000
    CosmicImage = np.zeros_like(data)
    total = 0
    sigma = np.std(data[data<1.2*np.median(data)])
    mean = np.median(data[data<1.2*np.median(data)])
    seeing = 5.
    while ((count) and (i <5)):
        count = LaplacianFilter(sigma, mean, seeing, data, CosmicImage)
        total+=count
        i+=1
    print(" Number of cosmics found ", count)
#    plt.imshow(CosmicImage,origin='lower')
#    plt.show()
    return CosmicImage

def LaplacianFilter(Sigma, Mean, seeing, data, CosmicImage):
#//Laplacian filter
#/*!Cuts (based on the article -> astro-ph/0108003):
#  -cut_lap : the laplacian operator increases the noise by a factor of
#  "sqrt(1.25)"
#
#  -cut_f : 2*sigma(med), where sigma(med) is the variance of the
#  sky's median calculated in a box (3*3), here.
#  (sigma(med) = sigma(sky)*1.22/sqrt(n); n = size of the box)
#
#  -cut_lf : calculated from the article.
#  Factor 2.35 -> to have the seeing in arc sec */
#  Adapted from code by A. Guyonnet
    xmax    = len(data)-1
    ymax    = len(data[0])-1
    l       = 0.0
    f       = 0.0
    med     = 0.0
    cut     = 4 * Sigma
    cut_lap = cut * np.sqrt(1.25)
    cut_f   = 2 * (Sigma*1.22/3)
    cut_lf  = 2./(seeing*2.35-1)
    count   = 0
    for j in range(1, ymax):
        for i in range(1, xmax):
            #Calculation of the laplacian and the median only for pixels > 3 sigma
            if (data[i,j] > cut+Mean ):
                l   = data[i,j] - 0.25*(data[i-1,j]   + data[i+1,j]
                                        + data[i,j-1] + data[i,j+1])
                med = np.median(data[i-1:i+2,j-1:j+2])
                f   = med - Mean  #f is invariant by addition of a constant
                #Construction of a cosmic image
                if((l>cut_lap) and ((f<cut_f) or ((l/f)>cut_lf))):
                   CosmicImage[i,j] = 1
                   data[i,j]        = med
                   count           += 1
            # Image is set to 0 by default
    #CosmicImage.Simplify(0.5)
    return count

def makeAsciiFile(waves,tpts,charge,fname):
    unique_waves = np.array(list(set(waves)))
    out = np.zeros_like(unique_waves)
    char = np.zeros_like(unique_waves)
    for i,w in enumerate(unique_waves):
        j = np.where(waves==w)[0]
        out[i] = np.median(tpts[j])
        char[i] = np.median(charge[j])
    np.savetxt(fname,np.column_stack((unique_waves,out,char)),header='WAVE RELTPT CHARGE')
def makeSpectrumPlot(xmlDict,nominalWave,fitsfilename):
    wave = xmlDict['spectrum']['wavelength']
    counts = xmlDict['spectrum']['counts']
    nominalWave = np.float(nominalWave)
    plt.figure(figsize=(12,12))
    plt.plot(wave,counts,'-k')
    plt.axvline(nominalWave,ls='--',color='r')
    plt.title('Nominal Wavelength: {}'.format(nominalWave))
    plt.xlabel('Wavelength [nm]',size=16)
    plt.ylabel('Counts',size=16)
    if not os.path.exists(os.path.join(os.path.dirname(fitsfilename),'diagnostics')):
        os.makedirs(os.path.join(os.path.dirname(fitsfilename),'diagnostics'))
    if not os.path.exists(os.path.join(os.path.dirname(fitsfilename),'diagnostics','spectrum')):
        os.makedirs(os.path.join(os.path.dirname(fitsfilename),'diagnostics','spectrum'))
    savepath = os.path.join(os.path.join(os.path.dirname(fitsfilename),'diagnostics','spectrum'),
                            os.path.split(fitsfilename[:-5])[-1]+'.png')
    plt.savefig(savepath)
    plt.clf()
    plt.close()
