
# coding: utf-8

# In[3]:

import os, sys
import optparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (13, 8) if False else (10, 6)

from astropy import wcs
import astropy.io.fits as pf
import PythonPhot as pp
import numpy as np
import scipy.interpolate

import xml.etree.ElementTree as ET

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-f","--logfile",default="/data/nas/CTIO_0p9m/october2017/logs_20170930/cbp_551.xml")
    parser.add_option("--doPlots",  action="store_true", default=False)

    opts, args = parser.parse_args()
  
    return opts

# Define a function (quadratic in our case) to fit the data with.
def lin_func(p, x):
    m, b = p
    return m*x + b

# Parse command line
opts = parse_commandline()
name = opts.logfile.split("/")[-1].replace(".xml","")

plotDir = 'output/%s'%(name)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

tree = ET.parse(opts.logfile)
doc = tree.getroot()

inst = doc.find("instrument_status")
angle, potentiometer_1, potentiometer_2, focus, filter, aperture, wavelength, x, y, z = inst.get('angle'), inst.get('potentiometer_1'), inst.get('potentiometer_2'), inst.get('focus'), inst.get('filter'), inst.get('aperture'), inst.get('wavelength'), inst.get('x'), inst.get('y'), inst.get('z')

times = []
datas = []
keithley = doc.find("keithley")
for elem in keithley.findall('keithley_element'):
    datas.append(float(elem.get('current'))) 
    times.append(float(elem.get('time'))) 

wavelengths = []
intensities = []
spectrograph = doc.find("spectrograph")
for elem in spectrograph.findall('spectrum'):
    wavelengths.append(float(elem.get('wavelength'))) 
    intensities.append(float(elem.get('intensity'))) 

times = np.array(times)
datas = np.abs(np.array(datas))
wavelengths = np.array(wavelengths)
intensities = np.array(intensities)

dataserr1 = np.abs(datas[4]-datas[0])
dataserr2 = np.abs(datas[-5]-datas[-1])
dataserr = (dataserr1+dataserr2)/2.0
datasint = np.abs(datas[-11]-datas[10])

timeserr1 = np.abs(times[4]-times[0])
timeserr2 = np.abs(times[-5]-times[-1])
timeserr = (timeserr1+timeserr2)/2.0
timesint = np.abs(times[-11]-times[10])

dataserrint = (timesint/timeserr)*dataserr

if opts.doPlots:
    plt.figure()
    plt.plot(times,datas*1e9,'kx')
    plt.xlabel("Time [s]")
    plt.ylabel("Integrated charge [nC]")
    plt.ylim([-0.5,5.5])
    plotName = os.path.join(plotDir,'timeseries.png')
    plt.savefig(plotName)
    plotName = os.path.join(plotDir,'timeseries.eps')
    plt.savefig(plotName)
    plotName = os.path.join(plotDir,'timeseries.pdf')
    plt.savefig(plotName)
    plt.close()
    
print "Cumulative charge: %.5e"%datas[-1]

photonfile = os.path.join(plotDir,'photons.dat')
fid = open(photonfile,'w')
fid.write('%.5e %.5e\n'%(datasint,dataserrint))
fid.close()

