import numpy as np
import glob
import astropy.io.fits as pft
import os

def makeChargeFile(direc):
    flist = glob.glob('C:\\Users\\Nick\\Documents\\CTIO_CBP\\Nofilter_405_955_2\\*.fits')
    fname = os.path.split(direc)[-1] + '_charge.txt'
    #with open(os.path.join(direc,fname),'w') as myfile:
    for f in flist:
        d = pft.open(f)
        hdr = d[0].header
        hdr_comment = hdr['COMMENT'][2]
        print(hdr['COMMENT'])
        wavelength = hdr_comment.split(' ')[1]