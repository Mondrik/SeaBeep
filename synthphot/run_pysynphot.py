#!/usr/bin/python

# Copyright (C) 2017 Michael Coughlin
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Synthetic photometry script.

This script uses collimated beam projector and atmospheric transmission
files to perform synthetic photometry on CALSPEC stars.

Comments should be e-mailed to michael.coughlin@ligo.org.

"""

import os, sys, optparse
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

__author__ = "Michael Coughlin <michael.coughlin@ligo.org>"
__version__ = 1.0
__date__    = "9/22/2013"

# =============================================================================
#
#                               DEFINITIONS
#
# =============================================================================

def parse_commandline():
    """@parse the options given on the command-line.
    """
    parser = optparse.OptionParser(usage=__doc__,version=__version__)

    parser.add_option("-p", "--plotDir", help="Plot directory.",
                      default ="../plots")
    parser.add_option("-d", "--dataDir", help="Data directory.",
                      default ="../data")
    parser.add_option("-s", "--star", help="star.",
                      default ="hd14943")
    parser.add_option("-f", "--filters", help="filter.",
                      default ="SLOAN-SDSS.g,SLOAN-SDSS.r,SLOAN-SDSS.i")
    parser.add_option("-a", "--atmosphere", help="atmosphere.",
                      default ="Tatmo_1")

    parser.add_option("-v", "--verbose", action="store_true", default=False,
                      help="Run verbosely. (Default: False)")

    opts, args = parser.parse_args()

    # show parameters
    if opts.verbose:
        print >> sys.stderr, ""
        print >> sys.stderr, "running run_pysynphot..."
        print >> sys.stderr, "version: %s"%__version__
        print >> sys.stderr, ""
        print >> sys.stderr, "***************** PARAMETERS ********************"
        for o in opts.__dict__.items():
          print >> sys.stderr, o[0]+":"
          print >> sys.stderr, o[1]
        print >> sys.stderr, ""

    return opts

# Parse command line
opts = parse_commandline()

plotDir = opts.plotDir
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

atmosfile = os.path.join(opts.dataDir,'atmosphere','%s.list'%opts.atmosphere)
atmos = np.loadtxt(atmosfile,comments='@',skiprows=25)
atmos[:,1] = atmos[:,1]/np.max(atmos[:,1])
atmosband = S.ArrayBandpass((atmos[:,0] * u.angstrom).value, np.clip(atmos[:,1], 0, np.inf), name='atmos')

bandpass_names = opts.filters.split(",")
bandpasses = []
for bandpass_name in bandpass_names:
    filename = "%s/instrument/%s.dat"%(opts.dataDir,bandpass_name)
    table = astropy.table.Table.read(filename, format='ascii', names=['wavelength', 'transmission'])
    band = S.ArrayBandpass((table['wavelength'] * u.angstrom).value, np.clip(table['transmission'], 0, np.inf), name=bandpass_name)
    #bandpasses.append(band*atmosband)
    bandpasses.append(band)

colors = sns.color_palette('spectral', len(bandpasses))

plt.figure(figsize=(10, 6))
for bandpass_name, bandpass, color in zip(bandpass_names, bandpasses, colors):
    plt.fill_between(bandpass.wave, bandpass.throughput, color=color, label=bandpass_name)
plt.legend(loc='upper left')
plotName = "%s/throughput.pdf"%plotDir
plt.savefig(plotName)
plt.close()

calspecfile = os.path.join(os.environ['PYSYN_CDBS'], 'grid', 'calspec',"%s_mod_001.fits"%opts.star)
spec = S.FileSpectrum(calspecfile)

A = (4 * np.pi * (100*u.Mpc)**2).cgs.value
wave = spec.wave
#flam = spec.flux / A
#spec = S.ArraySpectrum(wave, flam, 'angstrom', 'flam')

for bandpass in bandpasses:
    obs = S.Observation(spec, bandpass)
    ret = obs.effstim('abmag') - S.Observation(S.FlatSpectrum(0, fluxunits='abmag'), bandpass).effstim('abmag')

    print bandpass, ret

