import numpy as np
import astropy.io.fits as pft
import photutils as pt
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import os
import pickle
import cbphelpers as cbph
    
#important parameters
ap_phot_rad = 40
sky_rad_in = 45
sky_rad_out = 50
search_rad = 10

#file_list = glob.glob('20171005_I_scan_1/cbp_*.fits')
#combine = np.loadtxt('I_20171005_1_charge.txt',usecols=[2])
#pklfilename = '20171005_I_scan_1.pkl'
#useflat=True
#flatname = 'Iflat.fits'
#tpt_plot_name = '20171005_I_scan_1.png'
#info_dict = {}
#info_dict['dot_locs'] = np.loadtxt('spot_locs/20171005_I_scan_1.txt')
#info_dict['charge'] = combine

#file_list = glob.glob('Andover_scan_20171006_1/cbp_*.fits')
#combine = np.loadtxt('Andover_scan_20171006_1_charge.txt',usecols=[2])
#pklfilename = 'Andover_scan_20171006_1.pkl'
#useflat=True
#flatname = 'andoverflat.fits'
#tpt_plot_name = 'Andover_scan_20171006_1.png'
#info_dict = {}
#info_dict['dot_locs'] = np.loadtxt('spot_locs/Andover_scan_20171006_1.txt')
#info_dict['charge'] = combine


         
#file_list = glob.glob('Andover_scan_20171006_2_spotmove/cbp_*.fits')
#combine = np.loadtxt('Andover_scan_20171006_2_spotmove_charge.txt',usecols=[2])
#pklfilename = 'Andover_scan_20171006_2_spotmove.pkl'
#useflat=True
#flatname = 'andoverflat.fits'
#tpt_plot_name = 'Andover_scan_20171006_2_spotmove.png'
#info_dict = {}
#info_dict['dot_locs'] = np.loadtxt('spot_locs/Andover_scan_20171006_2_spotmove.txt')
#info_dict['charge'] = combine

file_list = glob.glob('20171006_RONCHI400_clear/cbp_*.fits')
combine = np.loadtxt('RONCHI400_clear_charge.txt',usecols=[2])
pklfilename = '20171006_RONCHI400_clear.pkl'
useflat=False
ronchi=True
can_move=False
flatname = 'andoverflat.fits'
tpt_plot_name = '20171006_RONCHI400_clear.png'
info_dict = {}
info_dict['dot_locs'] = np.loadtxt('spot_locs/20171006_RONCHI400_clear.txt')
info_dict['charge'] = combine
         
         
#file_list = glob.glob('Andover_scan_20171007_1/cbp_*.fits')
#combine = np.loadtxt('Andover_scan_20171007_1_charge.txt',usecols=[2])
#pklfilename = 'Andover_scan_20171007_1.pkl'
#useflat=True
#ronchi=False
#can_move=False
#flatname = 'andoverflat.fits'
#tpt_plot_name = 'Andover_scan_20171007_1.png'
#info_dict = {}
#info_dict['dot_locs'] = np.loadtxt('spot_locs/Andover_scan_20171007_1.txt')
#info_dict['charge'] = combine
#
#file_list = glob.glob('20171007_RONCHI400_obf/cbp_*.fits')
#combine = np.loadtxt('20171007_RONCHI400_obf_charge.txt',usecols=[2])
#pklfilename = '20171007_RONCHI400_obf.pkl'
#useflat=False
#ronchi=True
#can_move=False
#flatname = 'andoverflat.fits'
#tpt_plot_name = '20171007_RONCHI400_obf.png'
#info_dict = {}
#info_dict['dot_locs'] = np.loadtxt('spot_locs/20171007_RONCHI400_obf.txt')
#info_dict['charge'] = combine
#         
#file_list = glob.glob('I_scan_20171007/cbp_*.fits')
#combine = np.loadtxt('I_scan_20171007_charge.txt',usecols=[2])
#pklfilename = 'I_scan_20171007.pkl'
#useflat=False
#ronchi=False
#can_move=True
#flatname = 'Iflat.fits'
#tpt_plot_name = 'I_scan_20171007.png'
#info_dict = {}
#info_dict['dot_locs'] = np.loadtxt('spot_locs/I_scan_20171007.txt')
#info_dict['charge'] = combine

#file_list = glob.glob('20171011_nofilter_fullscan/cbp_*.fits')
#combine = np.loadtxt('20171011_nofilter_fullscan_charge.txt',usecols=[2])
#pklfilename = '20171011_nofilter_fullscan.pkl'
#useflat=True
#ronchi=False
#can_move=True
#flatname = 'whiteflat.fits'
#tpt_plot_name = '20171011_nofilter_fullscan.png'
#info_dict = {}
#info_dict['dot_locs'] = np.loadtxt('spot_locs/20171011_nofilter_fullscan.txt')
#info_dict['charge'] = combine

          
#file_list = glob.glob('20171009_I_redo_begin/cbp_*.fits')
#combine = np.loadtxt('20171009_I_redo_begin_charge.txt',usecols=[2])
#pklfilename = '20171009_I_redo_begin.pkl'
#useflat=True
#ronchi=False
#can_move=True
#flatname = 'Iflat.fits'
#tpt_plot_name = '20171009_I_redo_begin.png'
#info_dict = {}
#info_dict['dot_locs'] = np.loadtxt('spot_locs/20171009_I_redo_begin.txt')
#info_dict['charge'] = combine
#          

#file_list = glob.glob('20171011_nofilter_fullscan/cbp_*.fits')
#combine = np.loadtxt('20171011_nofilter_fullscan_charge.txt',usecols=[2])
#pklfilename = '20171011_nofilter_fullscan_noflat.pkl'
#useflat=False
#ronchi=False
#can_move=True
#flatname = 'whiteflat.fits'
#tpt_plot_name = '20171011_nofilter_fullscan_noflat.png'
#info_dict = {}
#info_dict['dot_locs'] = np.loadtxt('spot_locs/20171011_nofilter_fullscan.txt')
#info_dict['charge'] = combine

          
#make sure our cbp input file list is sorted in the same way as the combine.dat files
sort_arr = [float(i.split('_')[-1].split('.')[0]) for i in file_list]
sort_idxs = np.argsort(sort_arr)
file_list = np.array(file_list)
assert(len(info_dict['charge']) == len(file_list)),'Mismatched files...'


for i,dot in enumerate(info_dict['dot_locs']):
    info_dict['dot%d' % i] = {}
    info_dict['dot%d' % i]['flux'] = []
    info_dict['dot%d' % i]['dot_loc'] = []
    info_dict['dot%d' % i]['bkg_mean'] = []
    info_dict['dot%d' % i]['aper_uncert'] = []
    info_dict['dot%d' % i]['reltpt_uncert'] = []
    info_dict['dot%d' % i]['raw_tpt'] = []
    
info_dict['filename'] = []
info_dict['wavelengths'] = []
info_dict['exp_times'] = []
info_dict['bkg'] = []

if useflat:
    flatd = pft.open(flatname)
    flat = flatd[0].data * 3. #for gain from fits header.

for f in file_list[sort_idxs]:
    #====================================================
    #File IO
    print(f)
    info_dict['filename'].append(f)
    d = pft.open(f)
    hdr_comment = d[0].header['COMMENT'][2]
    wavelength = hdr_comment.split(' ')[1]
    print(wavelength)
    time = hdr_comment.split(' ')[3]
    shutter_stat = hdr_comment.split(' ')[-1]
    if shutter_stat == 'closed':
        continue
    data = d[0].data.copy() * 3. #3 for gain from fits header
    if useflat:
        data = data / flat

    info_dict['wavelengths'].append(np.float(wavelength))
    info_dict['exp_times'].append(np.float(time))
    info_dict['bkg'].append(np.median(data))
    #====================================================
    
    #====================================================
    #center location
    if can_move:
        if not ronchi:
            new_locs = []
            for i,pair in enumerate(info_dict['dot_locs']):
                myrow,mycol = cbph.findCenter(data,pair,search_size=search_rad)
                new_locs.append([myrow,mycol])  
            new_locs = np.array(new_locs)
            vec = new_locs - info_dict['dot_locs']
            vec = np.rint(np.median(vec,axis=0))
            new_locs = info_dict['dot_locs'] + vec
        else:
            new_locs = info_dict['dot_locs']
            myrow,mycol = cbph.findCenter(data,new_locs[0],search_size=search_rad)
            new_locs[0] = [myrow,mycol]
            new_locs[1] = new_locs[0]
            new_locs[1][1] += np.float(wavelength)*0.52769 - 17.61538
    else:
        if not ronchi:
            new_locs = info_dict['dot_locs']
        else:
            new_locs = info_dict['dot_locs']
            new_locs[1] = new_locs[0]
            new_locs[1][1] += np.float(wavelength)*0.52769 - 17.61538
    #=====================================================
        
    #=====================================================
    #Aperture photometry
    phot_table, bkg, error = cbph.do_aperture_photometry(new_locs,data,rad=ap_phot_rad,
                                                         sky_rad_in=sky_rad_in,sky_rad_out=sky_rad_out)
        
    for i in range(len(info_dict['dot_locs'])):
        info_dict['dot%d' % i]['dot_loc'].append(new_locs[i])
                
    for i,item in enumerate(phot_table['residual_aperture_sum']):
        info_dict['dot%d' % i]['flux'].append(item)
        info_dict['dot%d' % i]['bkg_mean'].append(bkg[i])
        info_dict['dot%d' % i]['aper_uncert'].append(error[i])
    #====================================================
    
    
    #====================================================
    #PLOTTING
    plt.figure(figsize=(12,12))
    plt.imshow(data-np.nanmedian(data.flatten()),origin='lower',vmin=-100,vmax=1000)
    
    #figure out where to put circles
    locs = new_locs

    wedges = []
    for j,loc in enumerate(locs):
        wedge = patch.Wedge(center=loc[::-1],theta1=0,theta2=360.,
                                r=sky_rad_out,width=sky_rad_out-sky_rad_in,
                                color='r',alpha=0.5)
        wedges.append(wedge)
        plt.text(loc[1]+sky_rad_out+10,loc[0]-sky_rad_out-10,'%d' % j,
                 color='w',size=16)
    for wedge in wedges:
        plt.gca().add_patch(wedge)

    circs = []
    for loc in locs:
        circ = patch.Circle(xy=loc[::-1],
                                radius=ap_phot_rad,color='k',alpha=0.5)
        circs.append(circ)
    for circ in circs:
        plt.gca().add_patch(circ)

    plt.colorbar(orientation='horizontal')
    plt.title(f)
    plt.text(800,100,'%snm' % wavelength,color='w',size=16)
    plt.savefig(f[:-5]+'.png')
    plt.clf()
    plt.close()
    plt.show()
    #====================================================
    info_dict['dot_locs'] = new_locs
#========================================================


info_dict['wavelengths'] = np.asarray(info_dict['wavelengths'])
info_dict['exp_times'] = np.asarray(info_dict['exp_times'])

plt.figure(figsize=(12,12))
cut_degen = False
for i in range(len(info_dict['dot_locs'])):
    x = np.array(info_dict['dot%d' % i]['flux']) / combine
    info_dict['dot%d' % i]['raw_tpt'].append(x)
    info_dict['dot%d' % i]['rel_tpt'] = x/np.max(x[info_dict['wavelengths']>=430])
    wavelength = info_dict['wavelengths']
    s = np.argsort(wavelength)
    yerr = np.array(info_dict['dot%d'%i]['aper_uncert']) / combine / np.max(x)
    for err in yerr:
        info_dict['dot%d' % i]['reltpt_uncert'].append(err)
    info_dict['dot%d' % i]['reltpt_uncert'] = np.asarray(info_dict['dot%d' % i]['reltpt_uncert'])
    yerr = np.asarray(yerr)
    if cut_degen:
        cut = np.logical_or(wavelength[s]<690,wavelength[s]>750)
    else:
        cut = wavelength[s]>0 #always true
    plt.errorbar(wavelength[s][cut],x[s][cut]/np.max(x),yerr=yerr[s][cut],marker='o',
                 label='%d'%i,ls='-',capsize=3,ms=3)
    
#a = np.loadtxt('450FSX40.csv',delimiter=',')
#plt.plot(a[:,0],a[:,1]/np.max(a[:,1]),'-k',lw=2)

plt.ylim(0,1.1)
plt.axhline(y=0,ls='--') 
plt.legend(ncol=2)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Relative Throughput')
plt.savefig(tpt_plot_name)
plt.show()

if ronchi:
    plt.errorbar(wavelength[s],info_dict['dot1']['rel_tpt'][s],
                 yerr=info_dict['dot1']['reltpt_uncert'][s],marker='o',
                 ls='-',capsize=3,ms=3)
    plt.ylim(0,1.1)
    plt.axhline(y=0,ls='--') 
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Relative Throughput')
    plt.show()


with open(pklfilename,'wb') as myfile:
    pickle.dump(info_dict,myfile)

#cbph.get_GP_model(info_dict)
#cbph.make_aux_plots(wavelength,combine,info_dict)