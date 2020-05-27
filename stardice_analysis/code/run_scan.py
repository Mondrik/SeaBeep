import CbpConfig as cc
import CbpProcessExposure as cpe
import CbpScan as cs
import os


scan_dir = '/home/mondrik/CBP/paris_data/empty_only_combination_scan'
base_name = os.path.split(scan_dir)[-1]

config = cc.CBPConfig()
config.data_location = scan_dir
config.log_file_name = os.path.join(scan_dir, base_name+'_runlog.log')

config.process_spectra = True
config.make_diagnostics = False

scan = cs.CbpScan(config)

scan.run_scan()

save_name = os.path.join(scan_dir, base_name+'_results.pkl')
scan.save(save_name)

