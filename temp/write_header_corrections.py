#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:50:44 2019

@author: ggilmore
"""
import os
os.chdir(r'C:\Users\Greydon\Documents\GitHub\software_solutions\edf2bids\main')
from helpers import EDFReader, padtrim

#%%

data_dir = r'G:\EDF_Studies\test\sub-017'
isub = 'sub-017'

files = [x for x in os.listdir(data_dir)]
for ifile in files:
	fname = os.path.join(data_dir, ifile)
	file_in = EDFReader()
	file_in.open(fname)
	header = file_in.readHeader()
	file_in.close()
	
	edf_subject_id = []
	recording_id = []
	if not header[0]['birthdate'] == 'X':
		edf_subject_id = 'X X X ' + isub
		
	if not header[0]['gender'] == 'X':
		edf_subject_id = 'X X X ' + isub
		
	if not header[0]['subject_id'] == isub:
		edf_subject_id = 'X X X ' + isub
	
	if not header[0]['recording_id'] == 'Startdate X X X X':
		recording_id = 'Startdate X X X X'
	
	
	with open(fname, 'r+b') as fid:
		assert(fid.tell() == 0)
		fid.seek(8)
		
		if edf_subject_id:
			fid.write(padtrim(edf_subject_id, 80).encode('utf-8'))
		else:
			fid.seek(80+8)
			assert(fid.tell() == 80+8)
		
		if recording_id:
			fid.write(padtrim(recording_id, 80).encode('utf-8'))
		else:
			fid.seek(80+80+8)
			assert(fid.tell() == 80+80+8)
		