
# coding: utf-8

# In[1]:

import os
os.environ['PYSYN_CDBS'] = os.path.abspath('../pysynphot_cdbs')
import cPickle as pickle
import numpy as np
import pysynphot as S
import astropy.units as u
import astropy.table
from astropy.utils.console import ProgressBar
from astropy.utils.data import download_file
import functools
import seaborn.apionly as sns 

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

bandpass_names = ["g","r","i"]
bandpasses = []
for bandpass_name in bandpass_names:
    filename = "input/SLOAN-SDSS.%s.dat"%bandpass_name
    table = astropy.table.Table.read(filename, format='ascii', names=['wavelength', 'transmission'])
    band = S.ArrayBandpass((table['wavelength'] * u.angstrom).value, np.clip(table['transmission'], 0, np.inf), name=bandpass_name)
    bandpasses.append(band)

colors = sns.color_palette('spectral', len(bandpasses))

plt.figure(figsize=(10, 6))
for bandpass_name, bandpass, color in zip(bandpass_names, bandpasses, colors):
    plt.fill_between(bandpass.wave, bandpass.throughput, color=color, label=bandpass_name)
plt.legend(loc='upper left')
plt.savefig('plots/throughput.pdf')
plt.close()

calspecfile = "hd14943_mod_001.fits"
spec = S.FileSpectrum(os.path.join(os.environ['PYSYN_CDBS'], 'grid', 'calspec', calspecfile))

A = (4 * np.pi * (100*u.Mpc)**2).cgs.value
wave = spec.wave
#flam = spec.flux / A
#spec = S.ArraySpectrum(wave, flam, 'angstrom', 'flam')

for bandpass in bandpasses:
    obs = S.Observation(spec, bandpass)
    ret = obs.effstim('abmag') - S.Observation(S.FlatSpectrum(0, fluxunits='abmag'), bandpass).effstim('abmag')

    print ret

