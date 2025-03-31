# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 19:19:24 2019

@author: greydon
"""
import os
import numpy as np
from collections import OrderedDict
import pandas as pd

from helpers import EDFReader, padtrim, sorted_nicely, partition, determine_groups


# Channel description for the recording equipment
natus_channel_info = {
                'DC': {'type': 'DAC', 'name': 'DAC'},
                'TRIG': {'type': 'TRIG', 'name': 'TR'},
                'OSAT': {'type': 'OSAT', 'name': 'OSAT'},
                'PR': {'type': 'PR', 'name': 'MISC'},
                'Pleth': {'type': 'Pleth', 'name': 'MISC'},
                'EDF Annotations': {'type': 'Annotations', 'name': 'ANNO'},
                'C': {'type': 'unused', 'name': 'unused'}
                }

def _participants_data(subject_id, file_info_sub, participants_fname):
    """
    Participant tsv file generation.
    
    """
    if subject_id is None:
        with open(participants_fname, 'w') as writeFile:
            writeFile.write("\t".join(['participant_id','age','sex','group']))
            writeFile.write( "\n" )
            
    else:
        # Parse edf files for age and gender (not all files will contain this info)
        age = []
        gender = []
        for ifile in file_info_sub:
            if isinstance(ifile['Age'], int):
                age = ifile['Age']
            if isinstance(ifile['Gender'], str):
                gender = ifile['Gender']

        df = pd.DataFrame(OrderedDict([
                          ('participant_id', 'sub-' + subject_id if subject_id is not None else ''),
                          ('age', age if age else 'n/a'),
                          ('sex', gender if gender else 'n/a'),
                          ('group', 'patient')]), index= [0])
        
        _write_tsv(participants_fname, df, overwrite=False, verbose=False, append = True) 
		
def get_file_info(raw_file_path):
    """
    Extract header data from EDF file.
    
    """
    file_list = [x for x in os.listdir(raw_file_path) if x.endswith('.edf')]
    filesInfo = []
    for ifile in file_list:
        filen = os.path.join(raw_file_path, ifile)
        file_in = EDFReader()
        try:
            meas_info, chan_info = file_in.open(filen)
            header = file_in.readHeader()
            file_in.close()
            
            values = partition(header[1]['ch_names'])
            seeg_chan_idx = [i for i, x in enumerate(values) if x[0] not in list(natus_channel_info.keys())]
            group_info = determine_groups(np.array(header[1]['ch_names'])[seeg_chan_idx])
            
            file_info = [
                ('FileName', ifile),
                ('Gender', meas_info['gender']),
                ('Age', int(np.floor(((datetime.datetime.strptime(meas_info['meas_date'].split(' ')[0],"%Y-%m-%d") 
                        - datetime.datetime.strptime(datetime.datetime.strptime(meas_info['birthdate'],
                        '%d-%b-%Y').strftime('%Y-%m-%d'),"%Y-%m-%d")).days)/365)) if meas_info['birthdate'] else ''),
                ('Date', meas_info['meas_date'].split(' ')[0]),
                ('Time', meas_info['meas_date'].split(' ')[1]),
                ('DataOffset', meas_info['data_offset']),
                ('NRecords', meas_info['n_records']),
                ('RecordLength', meas_info['record_length']),
                ('NChan', meas_info['nchan']),
                ('Highpass', meas_info['highpass']),
                ('Lowpass', meas_info['lowpass']),
                ('Groups', group_info)]
            
            ch_info = {}
            ch_info['SEEG'] = OrderedDict([
                ('ChannelCount', len(seeg_chan_idx)),
                ('Unit', np.array(header[1]['units'])[seeg_chan_idx]),
                ('ChanName', np.array(header[1]['ch_names'])[seeg_chan_idx]),
                ('Type', 'SEEG')])
            
            for key in natus_channel_info.keys():
                chan_idx = [i for i, x in enumerate(values) if x[0] == key]
                ch_info[key] = OrderedDict([
                        ('ChannelCount', len(chan_idx)),
                        ('Unit', np.array(header[1]['units'])[chan_idx]),
                        ('ChanName', np.array(header[1]['ch_names'])[chan_idx]),
                        ('Type', natus_channel_info[key]['name'])])
            
            file_info.append(['ChanInfo',OrderedDict(ch_info)])
            
            file_info = OrderedDict(file_info)
            file_info['ChanInfo'] = ch_info
            filesInfo.append(file_info)
        except:
            print('Something wrong with file', ifile)
        
    filesInfo = sorted(filesInfo, key=lambda k: (k['Date'], k['Time'])) 
    
    return filesInfo

def read_input_dir(input_dir):
	folders = [x for x in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir,x))]
	sub_file_info = {}
	for ifold in folders:
		subject_id = ifold
		raw_file_path = os.path.join(input_dir, ifold)
		file_info = get_file_info(raw_file_path)
		
		sub_file_info[subject_id] = file_info
		
	return sub_file_info

input_path = r'/media/veracrypt1/projects/eplink/lhs_data/test'
output_path = r'/media/veracrypt1/projects/eplink/lhs_data/test_out'
file_info = read_input_dir(input_path)
new_sessions = read_output_dir(output_path, file_info)

subject_id = 'sub-P009'
_participants_data(isub, file_info, participants_fname)

participants_fname = r'F:/projects/iEEG/out/participants.tsv'
file_info_sub = file_info[isub]
def _participants_data(self, subject_id, file_info_sub, participants_fname):
	"""
	Participant tsv file generation.
	
	"""
	if subject_id is None:
		with open(participants_fname, 'w') as writeFile:
			writeFile.write("\t".join(['participant_id','age','sex','group']))
			writeFile.write( "\n" )
			
	else:
		# Parse edf files for age and gender (not all files will contain this info)
		age = []
		gender = []
		for ifile in file_info_sub:
			if isinstance(ifile['Age'], int):
				age = ifile['Age']
			if isinstance(ifile['Gender'], str):
				gender = ifile['Gender']

		df = pd.DataFrame(OrderedDict([
						  ('participant_id', subject_id if subject_id is not None else ''),
						  ('age', age if age else 'n/a'),
						  ('sex', gender if gender else 'n/a'),
						  ('group', 'patient')]), index= [0])
	return df

def read_output_dir(output_path, file_info):
	folders = [x for x in os.listdir(output_path) if os.path.isdir(os.path.join(output_path,x)) and 'code' not in x]
	new_sessions = {}
	for isub, values in file_info.items():
		num_sessions = len(values)
		newSessions = True
		sub_sessions = {}
		
		# True if subject directory already in output directory
		if isub in folders:
			sessionsDone = sorted_nicely([x for x in os.listdir(os.path.join(output_path, isub)) if x.startswith('ses')])
			
			# Determine if new sessions to run, and where to begin
			if os.path.exists(os.path.join(output_path, isub, isub + '_scans.tsv')):
				# Load output scan file
				scans_tsv = pd.read_csv(os.path.join(output_path, isub, isub + '_scans.tsv'), sep='\t')
				
				# Extract time and dates from input files
				session_dates = [x['Date']+'T'+x['Time'] for x in values]
				
				# Find input sessions already in output directory
				unique_sessions = [x for x,i in enumerate(session_dates) if i in list(scans_tsv['acq_time'])]
				
				# Extract input session values not in output directory
				values = [i for j, i in enumerate(values) if j not in unique_sessions]
				
				if not values:
					newSessions = False
					session_start = np.nan
				else:
					# Determine that session number to start from with new session files
					session_start = len(sessionsDone)
					num_sessions = session_start + len(values)
		else:
			session_start = 0
			
		sub_sessions['newSessions'] = newSessions
		sub_sessions['session_start'] = session_start
		sub_sessions['num_sessions'] = num_sessions
		
		new_sessions[isub] = sub_sessions
		
	return new_sessions
			
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		