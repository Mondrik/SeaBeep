import numpy as np
import photutils as pt
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import os
import astropy.io.fits as pft
import logging


def make_directories(name, fitsfilename, base_folder='diagnostics'):
    if not os.path.exists(os.path.join(os.path.dirname(fitsfilename), 'diagnostics')):
        os.makedirs(os.path.join(os.path.dirname(fitsfilename), 'diagnostics'))
    if not os.path.exists(os.path.join(os.path.dirname(fitsfilename), 'diagnostics', name)):
        os.makedirs(os.path.join(os.path.dirname(fitsfilename), 'diagnostics', name))


def make_images(config, data, results, dark_data):
    locs = results.dot_loc
    wavelength = results.laser_wavelength
    fitsfilename = results.file_name

    #  Plot an image of the entire data frame, with circles drawn on aperture and sky aperture locations
    plt.ioff()
    plt.figure(figsize=(12, 12))
    vmin, vmax = config.whole_frame_minmax
    plt.imshow(data, origin='lower', vmin=vmin, vmax=vmax)
    wedges = []

    for j, loc in enumerate(locs):
        wedge = patch.Wedge(center=loc[::-1],theta1=0, theta2=360.,
                            r=config.sky_rad_out, width=config.sky_rad_out-config.sky_rad_in,
                            ec='r', fc=None, lw=2)
        wedges.append(wedge)
        plt.text(loc[1]+config.sky_rad_out+10, loc[0]-config.sky_rad_out-10, '%d' % j,
                 color='w', size=config.label_size)
    for wedge in wedges:
        plt.gca().add_patch(wedge)

    circs = []
    for loc in locs:
        circ = patch.Circle(xy=loc[::-1],
                            radius=config.ap_phot_rad, ec='white', fc=None, lw=2)
        circs.append(circ)
    for circ in circs:
        plt.gca().add_patch(circ)

    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_label('Counts', size=config.label_size)
    plt.xlabel('X [pix]', size=config.label_size)
    plt.ylabel('Y [pix]', size=config.label_size)
    plt.text(625, 50, '%snm' % wavelength, color='w', size=config.label_size)
    make_directories('images', fitsfilename)
    savepath = os.path.join(os.path.join(os.path.dirname(fitsfilename), 'diagnostics', 'images'),
                            os.path.split(fitsfilename[:-5])[-1]+'.png')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.clf()
    plt.close()

    #  Plot an image of the dark frame
    if dark_data is not None:
        make_directories('darks', fitsfilename)
        savepath = os.path.join(os.path.join(os.path.dirname(fitsfilename), 'diagnostics', 'darks'),
                                os.path.split(fitsfilename[:-5])[-1]+'.png')
        plt.title('%snm' % wavelength)
        vmin, vmax = np.percentile(dark_data, [10,90])
        plt.imshow(dark_data, origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(orientation='horizontal')
        plt.tight_layout()
        plt.savefig(savepath)
        plt.clf()
        plt.close()
   

def make_spectrum_plot(config, results):
    laser_wavelength = results.laser_wavelength
    wavelengths = results.spectrum[:,0]
    counts = results.spectrum[:,1]
    fitsfilename = results.file_name

    plt.ioff()
    plt.figure(figsize=(12,12))
    plt.plot(wavelengths, counts, '-k,', label='Raw')
    if config.plot_pix2wave_spectrum:
        fit = np.loadtxt(config.pix2wave_fit_file)
        pix2wave = np.polynomial.chebyshev.Chebyshev(fit)
        pix = np.arange(0, len(wavelengths), 1)
        plt.plot(pix2wave(pix), counts, '-b', label='Calibrated')

    plt.axvline(laser_wavelength, ls='--', color='r', label='Requested')
    plt.legend()
    plt.title('Nominal Wavelength: {}'.format(laser_wavelength))
    plt.xlabel('Wavelength [nm]', size=config.label_size)
    plt.ylabel('Counts', size=config.label_size)
    plt.tight_layout()
    make_directories('spectrum', fitsfilename)
    savepath = os.path.join(os.path.join(os.path.dirname(fitsfilename), 'diagnostics', 'spectrum'),
                            os.path.split(fitsfilename[:-5])[-1]+'.png')
    plt.savefig(savepath)
    plt.clf()
    plt.close()

