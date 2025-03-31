#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:38:22 2019

@author: ggilmore
"""
import os
import pandas as pd
from collections import OrderedDict

from helpers import EDFReader, sorted_nicely, partition, determine_groups, fix_sessions, read_annotation_block, deidentify_edf, get_file_info, _write_tsv
from bids_settings import ieeg_file_metadata, natus_info

bids_settings = {}
bids_settings['json_metadata'] = ieeg_file_metadata()
bids_settings['natus_info'] = natus_info()
bids_settings['settings_panel'] = {'Deidentify_source': False,
									   'offset_dates': True}
			
raw_file_path = r'/home/ggilmore/Downloads/graham/projects/rrg-akhanf/cfmm-bids/Khan/epi_iEEG/bids'
ignore_subs = ['sub-018','sub-020','sub-021','sub-022', 'sub-023', 'sub-024', 'sub-025']
folders = sorted_nicely([x for x in os.listdir(raw_file_path) if os.path.isdir(os.path.join(raw_file_path, x)) and 'code' not in x and x not in ignore_subs])

for isub in folders:
	session_list = sorted_nicely([x for x in os.listdir(os.path.join(raw_file_path, isub)) if os.path.isdir(os.path.join(raw_file_path, isub, x))])
	# Load output scan file
	scans_tsv = pd.read_csv(os.path.join(raw_file_path, isub, isub + '_scans.tsv'), sep='\t')
	scans_update = []			
	for ises in session_list:
		sub_directory = [x for x in os.listdir(os.path.join(raw_file_path, isub, ises)) if os.path.isdir(os.path.join(raw_file_path, isub, ises, x))][0]
		session_dir = os.path.join(raw_file_path, isub, ises, sub_directory)
		file_info = get_file_info(session_dir, bids_settings)[0]
		for values in file_info:
			name_idx = [i for i,x in enumerate(list(scans_tsv['filename'])) if values['FileName'] in x]
			assert len(name_idx) == 1
			new_date = values['Date'] + 'T' + values['Time']
			
			scans_update.append(OrderedDict([('filename', values['FileName']),('acq_time', new_date),
			('duration', values['TotalRecordTime']),('edf_type', values['EDF_type'])]))
		
	df = pd.DataFrame(scans_update)
	_write_tsv(os.path.join(raw_file_path, isub, isub + '_scans.tsv'), df, overwrite=True, verbose=False, append = False)
			
