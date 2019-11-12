import numpy as np
import astropy.io.fits as pft
import glob
import matplotlib.pyplot as plt
import os
import pickle
import cbp_helpers as cbph
import time

def getStandardParams():
    params = {}
    params['ap_phot_rad'] = 45
    params['sky_rad_in'] = params['ap_phot_rad'] + 5
    params['sky_rad_out'] = params['sky_rad_in'] + 5
    params['search_rad'] = 10
    params['use_flat'] = False
    params['ronchi'] = True
    params['can_move'] = False
    params['gain'] = 3.
    params['use_overscan'] = False
    return params

def processCBP(params=None,fits_file_path=None,flat_directory=None,flat_name=None,
               make_plots=True,bkg_method='2d',suffix=''):
    startTime = time.time()
    #important parameters -- If not given, make assumptions
    if params is None:
        params = getStandardParams()
    
    #Inputs
    if fits_file_path is None:
        fits_file_path = 'C:\\Users\\Nick\\Documents\\CTIO_CBP\\20171007_RONCHI400_obf'
    if flat_directory is None:
        flat_directory = 'C:\\Users\\Nick\\Documents\\CTIO_CBP\\flats'
    if flat_name is None:
        flat_name = 'andover_flat.fits'
    
    
    #File location and checking
    root_name = os.path.split(fits_file_path)[-1]
    file_list = glob.glob(os.path.join(fits_file_path,'*.fits'))
    assert(len(file_list)>0),'No fits files found in %s' % fits_file_path
    
    flat_name = os.path.join(flat_directory,flat_name)
    if params['use_flat']:
        assert(os.path.exists(flat_name)),'Flat field %s no found!' % flat_name
    
    charge_file = os.path.join(fits_file_path,root_name+'_charge.txt')
    assert(os.path.exists(charge_file)),'Charge file %s not found!' % charge_file
    charge_data = np.atleast_1d(np.loadtxt(charge_file,usecols=[2]))
    
    spot_file = os.path.join(fits_file_path,root_name+'_spots.txt')
    assert(os.path.exists(spot_file)),'Spot file %s not found!' % spot_file
    dot_locs = np.loadtxt(spot_file)
    
    #Outputs
    #output files + plots
    pkl_filename = os.path.join(fits_file_path,root_name + suffix + '.pkl')
    tpt_plot_name = os.path.join(fits_file_path,root_name + suffix + '.png')
    
    
    #initialize input dictionary
    info_dict = {}
    info_dict['dot_locs'] = dot_locs
    info_dict['charge'] = charge_data
    info_dict['params'] = params
    info_dict['run_name'] = os.path.split(fits_file_path)[-1]
              
    #make sure our cbp input file list is sorted in the same way as the combine.dat files
    sort_arr = [float(i.split('_')[-1].split('.')[0]) for i in file_list]
    sort_idxs = np.argsort(sort_arr)
    file_list = np.array(file_list)
    assert(len(info_dict['charge']) == len(file_list)),'Mismatched files...'
    
    #set up our output dictionary
    for i,dot in enumerate(info_dict['dot_locs']):
        info_dict['dot%d' % i] = {}
        info_dict['dot%d' % i]['flux'] = []
        info_dict['dot%d' % i]['raw_flux'] = []
        info_dict['dot%d' % i]['dot_loc'] = []
        info_dict['dot%d' % i]['aper_uncert'] = []
    info_dict['filename'] = []
    info_dict['wavelengths'] = []
    info_dict['exp_times'] = []
    info_dict['wavelength_cal_linear'] = np.array([0.99811784,2.25080503])
    info_dict['wavelength_cal_cubic'] = np.array([1.01158924E-8,-2.29992990E-5,
             1.01457686E0,-1.39449844E0])
    
    if params['use_flat']:
        flatd = pft.open(flat_name)
        #cut off last 30 columns, as they are overscan
        flat = flatd[0].data[:,:-30]
    
    for fnum,f in enumerate(file_list[sort_idxs]):
        #====================================================
        #File IO + header parsing
        info_dict['filename'].append(f)
        d = pft.open(f)
        hdr_comment = d[0].header['COMMENT'][2]
        wavelength = hdr_comment.split(' ')[1]
        print(os.path.split(f)[-1],wavelength,'{} of {}'.format(fnum+1,len(file_list)))
        expTime = hdr_comment.split(' ')[3]
        shutter_stat = hdr_comment.split(' ')[-1]
        #we don't use the CBP off data
        if shutter_stat == 'closed':
            continue
        #cut off last 30 columns, as they are overscan
        overscan = np.median(d[0].data[:,-30:],axis=1) * params['gain']
        data = d[0].data[:,:-30] * params['gain']
        
        # Do some pre-processing of the data
        #cbph.filterCosmics(data)
        if params['use_overscan']:
            overscan_array = np.repeat(overscan[:,np.newaxis],data.shape[0],axis=1)
            data = data - overscan_array
        if params['use_flat']:
            data = data / flat
    
        info_dict['wavelengths'].append(np.float(wavelength))
        info_dict['exp_times'].append(np.float(expTime))
        #====================================================
        
        #====================================================
        #determine spot locations
        new_locs = cbph.getNewLocs(data, info_dict, params)
        #=====================================================
            
        #=====================================================
        #Aperture photometry
        phot_table, bkg, error = cbph.doAperturePhotometry(new_locs,data,f,params,bkg_method=bkg_method)
        #=====================================================
        
        #Update info dictionary with new photometry + locations
        #=====================================================
        for i in range(len(info_dict['dot_locs'])):
            info_dict['dot%d' % i]['dot_loc'].append(new_locs[i])
            info_dict['dot%d' % i]['flux'].append(phot_table['residual_aperture_sum'][i])
            info_dict['dot%d' % i]['raw_flux'].append(phot_table['aperture_sum'][i])
            info_dict['dot%d' % i]['aper_uncert'].append(error[i])
        info_dict['dot_locs'] = new_locs
        #====================================================
        
        #====================================================
        #PLOTTING
        cbph.makeDiagnosticPlots(data,new_locs,params,f,wavelength,bkg)
        if not params['ronchi']:
            cbph.makeDotHistograms(data,new_locs,10,f,wavelength,bkg)
            cbph.makeDotImages(data,new_locs,15,f,wavelength,bkg)
        xmlName = os.path.join(fits_file_path,
                               os.path.join('xml',os.path.split(f)[-1][:-5]+'.xml'))
        xmlDict = cbph.getXMLDict(xmlName)
        cbph.makeSpectrumPlot(xmlDict,wavelength,f)
        #====================================================
        
    #begin post-processing of photometry
    #========================================================
    #sort output quantites by wavelength
    info_dict['wavelengths'] = np.asarray(info_dict['wavelengths'])
    info_dict['exp_times'] = np.asarray(info_dict['exp_times'])
    s = np.argsort(info_dict['wavelengths'])
    
    #calculate throughputs
    for i,dot in enumerate(info_dict['dot_locs']):
        info_dict['dot%d' % i]['flux'] = np.asarray(info_dict['dot%d' % i]['flux'],dtype=np.float)[s]
        info_dict['dot%d' % i]['raw_flux'] = np.array(info_dict['dot%d' % i]['raw_flux'],dtype=np.float)[s]
        info_dict['dot%d' % i]['dot_loc'] = np.asarray(info_dict['dot%d' % i]['dot_loc'])[s]
        info_dict['dot%d' % i]['aper_uncert'] = np.asarray(info_dict['dot%d' % i]['aper_uncert'])[s]
    info_dict['filename'] = np.asarray(info_dict['filename'],dtype=np.str)[s]
    info_dict['wavelengths'] = info_dict['wavelengths'][s]
    info_dict['exp_times'] = info_dict['exp_times'][s]
    info_dict['charge'] = info_dict['charge'][s]
    charge_mask = info_dict['charge'] > 1E-7
    #==============================================
    
    if make_plots:
        plt.figure(figsize=(12,12))
    wavelength = info_dict['wavelengths']
    for i in range(len(info_dict['dot_locs'])):
        info_dict['charge_uncert'] = info_dict['charge']*0.01 + 50. * 1e-11
        x = info_dict['dot%d' % i]['flux'] / info_dict['charge']
        info_dict['dot%d' % i]['raw_tpt'] = np.asarray(x,dtype=np.float)
        info_dict['dot%d' % i]['rel_tpt'] = x/np.max(x[charge_mask])
#        yerr = info_dict['dot%d'%i]['aper_uncert'] / info_dict['charge'] / np.max(x[charge_mask])
        yerr = cbph.getTptUncert(info_dict['dot%d'%i]['aper_uncert'],info_dict['charge_uncert'],
                                 info_dict['dot%d'%i]['flux'],info_dict['charge'])
        info_dict['dot%d' % i]['rawtpt_uncert'] = yerr
        if make_plots:
            yerr = np.asarray(yerr)
            plt.errorbar(wavelength,x/np.max(x[charge_mask]),yerr=yerr/np.max(x[charge_mask]),marker='o',
                         label='%d'%i,ls='-',capsize=3,ms=3)
    

    with open(pkl_filename,'wb') as myfile:
        pickle.dump(info_dict,myfile)
        
    #make median throughput array + file:
    n = len(info_dict['dot_locs'])
    tptarr = np.zeros((len(wavelength),n)).astype(np.float)
    for i in range(len(info_dict['dot_locs'])):
        tptarr[:,i] = np.array(info_dict['dot%d'%i]['raw_tpt']) / np.max(info_dict['dot%d'%i]['raw_tpt'][charge_mask])
    tpts = np.median(tptarr,axis=1)
    asc_file_name = os.path.join(fits_file_path,root_name+suffix+'_median_tpt.txt')
    cbph.makeAsciiFile(wavelength,tpts,info_dict['charge'],fname=asc_file_name)
    
    if make_plots:
        plt.ylim(0,1.1)
        plt.axhline(y=0,ls='--') 
        plt.legend(ncol=2)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Relative Throughput')
        plt.savefig(tpt_plot_name)
        plt.show()
    
    print('Processing took {:6.1f} seconds.'.format(time.time()-startTime))
    return info_dict