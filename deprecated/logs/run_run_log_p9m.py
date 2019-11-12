
import os, sys, glob, fnmatch
import optparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (13, 8) if False else (10, 6)

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("--logDir",default="/data/nas/CTIO_0p9m/temp_oct17_data")
    parser.add_option("--doPlots",  action="store_true", default=False)

    opts, args = parser.parse_args()
 
    return opts

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

# Parse command line
opts = parse_commandline()
filename = os.path.join("output","combine.dat")

fid = open(filename,'w')
xmlfiles = find_files(opts.logDir, '*.xml')
for xmlfile in xmlfiles:
    foldername = xmlfile.split("/")[-2]
    name = xmlfile.split("/")[-1].replace(".xml","")
    if name == "": continue
    name = "%s_%s"%(foldername,name)
    #if True:
    if not os.path.isfile("output/%s/photons.dat"%(name)):
        if opts.doPlots:
            system_command = "python run_log_p9m.py -f %s --doPlots"%(xmlfile)
            print system_command
            os.system(system_command)
        else:
            system_command = "python run_log_p9m.py -f %s"%(xmlfile)
            print system_command
            os.system(system_command)

    data_out = np.loadtxt("output/%s/photons.dat"%(name)) 
    fid.write('%s %.5e %.5e\n'%(xmlfile,data_out[0],data_out[1]))
fid.close()

