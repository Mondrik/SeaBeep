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

"""Filter comparison script.

This script uses collimated beam projector and vendor transmission
files to compare results.

Comments should be e-mailed to michael.coughlin@ligo.org.

"""

import os, sys, optparse, re

if not os.getenv("DISPLAY", None):
    import matplotlib
    matplotlib.use("agg", warn=False)
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
else:
    import matplotlib
    matplotlib.use("agg", warn=False)
    import matplotlib.pyplot as plt 
    from matplotlib.pyplot import cm

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
    parser.add_option("-f", "--filters", help="filter.",
                      default = "r_filter_CBP,I_filter_CBP")
    parser.add_option("-r", "--reffilters", help="filter.",
                      default = "R_tek2,I_tek2")

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

bandpass_names = opts.filters.split(",")
bandpasses = []
for bandpass_name in bandpass_names:
    filename = "%s/instrument/%s.dat"%(opts.dataDir,bandpass_name)
    if "SDSS" in filename:
        table = astropy.table.Table.read(filename, format='ascii', names=['wavelength', 'transmission'])
        band = S.ArrayBandpass((table['wavelength'] * u.angstrom).value, np.clip(table['transmission'], 0, np.inf), name=bandpass_name)
    elif "tek2" in filename:
        table = astropy.table.Table.read(filename, format='ascii', names=['wavelength', 'transmission'])
        band = S.ArrayBandpass((10 * table['wavelength'] * u.angstrom).value, np.clip(table['transmission']/np.max(table['transmission']), 0, np.inf), name=bandpass_name)
    elif "I" in filename:
        table = astropy.table.Table.read(filename, format='ascii', names=['wavelength', 'transmission'])
        band = S.ArrayBandpass((10 * table['wavelength'] * u.angstrom).value, np.clip(table['transmission'], 0, np.inf), name=bandpass_name)
    else:
        table = astropy.table.Table.read(filename, format='ascii', names=['wavelength', 'transmission', 'charge'])
        band = S.ArrayBandpass((10 * table['wavelength'] * u.angstrom).value, np.clip(table['transmission'], 0, np.inf), name=bandpass_name)
    bandpasses.append(band)

refbandpass_names = opts.reffilters.split(",")
refbandpasses = []
for bandpass_name in refbandpass_names:
    filename = "%s/instrument/%s.dat"%(opts.dataDir,bandpass_name)
    if "SDSS" in filename:
        table = astropy.table.Table.read(filename, format='ascii', names=['wavelength', 'transmission'])
        band = S.ArrayBandpass((table['wavelength'] * u.angstrom).value, np.clip(table['transmission'], 0, np.inf), name=bandpass_name)
    elif "tek2" in filename:
        table = astropy.table.Table.read(filename, format='ascii', names=['wavelength', 'transmission'])
        band = S.ArrayBandpass((10 * table['wavelength'] * u.angstrom).value, np.clip(table['transmission']/np.max(table['transmission']), 0, np.inf), name=bandpass_name)
    elif "I" in filename:
        table = astropy.table.Table.read(filename, format='ascii', names=['wavelength', 'transmission'])
        band = S.ArrayBandpass((10 * table['wavelength'] * u.angstrom).value, np.clip(table['transmission'], 0, np.inf), name=bandpass_name)
    else:
        table = astropy.table.Table.read(filename, format='ascii', names=['wavelength', 'transmission', 'charge'])
        band = S.ArrayBandpass((10 * table['wavelength'] * u.angstrom).value, np.clip(table['transmission'], 0, np.inf), name=bandpass_name)
    refbandpasses.append(band)

colors = sns.color_palette('spectral', len(bandpasses))

plt.figure(figsize=(10, 6))
for bandpass_name, bandpass, color in zip(bandpass_names, bandpasses, colors):
    plt.plot(bandpass.wave, bandpass.throughput, '-', color=color, label=bandpass_name)
for bandpass_name, bandpass in zip(refbandpass_names, refbandpasses):
    plt.plot(bandpass.wave, bandpass.throughput, '--', color='k')
plt.legend(loc='best')
plotName = "%s/filter_comparison.pdf"%plotDir
plt.savefig(plotName)
plt.close()


