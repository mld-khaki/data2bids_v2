# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:15:53 2019

@author: User
"""
import os
import numpy as np
import pandas as pd
pd.set_option('precision', 6)
import json
from collections import OrderedDict
import datetime
import shutil
import re

from helpers import EDFReader, padtrim, sorted_nicely, partition, determine_groups, get_file_info
from bids_settings import ieeg_file_metadata, coord_system_info, natus_channel_info

ieeg_file_metadata = ieeg_file_metadata()
coord_system_info = coord_system_info()		
natus_channel_info = natus_channel_info()


raw_file_path = r'F:/projects/iEEG/sourcedata/sub-001/sub-001_SE03_IEEG_CLIP'
raw_file_path = r'/home/ggilmore/Downloads/graham/projects/rrg-akhanf/cfmm-bids/Khan/epi_iEEG/source_data/sub-015/from_suzan'

file_list = [x for x in os.listdir(raw_file_path) if x.lower().endswith('.edf')]
sub_dir = False
if not file_list:
	folders = [x for x in os.listdir(raw_file_path) if os.path.isdir(os.path.join(raw_file_path, x))]
	sub_dir=True
	for ifold in folders:
		files = [x for x in os.listdir(os.path.join(raw_file_path, ifold)) if x.lower().endswith('.edf')]
		if len(files)==1:
			file_list.append(os.path.sep.join([ifold, files[0]]))
				
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
			('DisplayName', ifile if not sub_dir else ifile.split(os.path.sep)[0]),
			('Subject', meas_info['subject_id']),
			('Gender', meas_info['gender']),
			('Age', int(np.floor(((datetime.datetime.strptime(meas_info['meas_date'].split(' ')[0],"%Y-%m-%d") 
					- datetime.datetime.strptime(datetime.datetime.strptime(meas_info['birthdate'],
					'%d-%b-%Y').strftime('%Y-%m-%d'),"%Y-%m-%d")).days)/365)) if meas_info['birthdate'] != 'X' else 'X'),
			('Birthdate', meas_info['birthdate']),
			('RecordingID', meas_info['recording_id']),
			('Date', meas_info['meas_date'].split(' ')[0]),
			('Time', meas_info['meas_date'].split(' ')[1]),
			('DataOffset', meas_info['data_offset']),
			('NRecords', meas_info['n_records']),
			('RecordLength', meas_info['record_length']),
			('TotalRecordTime', round((((meas_info['n_records']*(meas_info['sampling_frequency']*meas_info['record_length']))/meas_info['sampling_frequency'])/60)/60,3)),
			('NChan', meas_info['nchan']),
			('SampleFrequency', meas_info['sampling_frequency']),
			('Highpass', meas_info['highpass']),
			('Lowpass', meas_info['lowpass']),
			('Groups', group_info),
			('EDF_type', header[0]['subtype'])]
		
		if header[0]['subtype'] == 'EDF+D':
			print(ifile)
			
		file_info = OrderedDict(file_info)
				
		if file_info['DisplayName'].lower().endswith('.edf'):
			if file_info['NChan'] < 60:
				file_info['RecordingType'] = 'Scalp'
			else:
				file_info['RecordingType'] = 'iEEG'
			
			if file_info['TotalRecordTime'] < 5:
				file_info['RecordingLength'] = 'Clip'
			else:
				file_info['RecordingLength'] = 'Full'
		else:
			if 'ieeg' in file_info['DisplayName'].lower():
				file_info['RecordingType'] = 'iEEG'
			else:
				file_info['RecordingType'] = 'Scalp'
				
			if 'cs' in file_info['DisplayName'].lower():
				file_info['RecordingLength'] = 'CS'
			elif 'full' in file_info['DisplayName'].lower():
				file_info['RecordingLength'] = 'Full'
			elif 'clip' in file_info['DisplayName'].lower():
				file_info['RecordingLength'] = 'Clip'
		
		if file_info['RecordingType'] == 'Scalp':
			chan_type = 'EEG'
		elif file_info['RecordingType'] == 'iEEG':
			chan_type = 'SEEG'
			
		ch_info = {}
		ch_info[chan_type] = OrderedDict([
			('ChannelCount', len(seeg_chan_idx)),
			('Unit', np.array(header[1]['units'])[seeg_chan_idx]),
			('ChanName', np.array(header[1]['ch_names'])[seeg_chan_idx]),
			('Type', chan_type)])
		
		for key in natus_channel_info.keys():
			chan_idx = [i for i, x in enumerate(values) if x[0] == key]
			ch_info[key] = OrderedDict([
					('ChannelCount', len(chan_idx)),
					('Unit', np.array(header[1]['units'])[chan_idx]),
					('ChanName', np.array(header[1]['ch_names'])[chan_idx]),
					('Type', natus_channel_info[key]['name'])])
		
		file_info['ChanInfo'] = OrderedDict(ch_info)
		file_info = OrderedDict(file_info)
			
		filesInfo.append(file_info)
	except:
		print('Something wrong with file', ifile)
	
filesInfo = sorted(filesInfo, key=lambda k: (k['Date'], k['Time'])) 

file_data = filesInfo

def _write_tsv(fname, df, overwrite=False, verbose=False, append = False):
	if os.path.exists(fname) and not overwrite:
		pass
	if os.path.exists(fname) and append:
		with open(fname,'a') as f:
			df.to_csv(f, sep='\t', index=False, header = False, na_rep='n/a', mode='a', line_terminator="")
		with open(fname) as f:
			lines = f.readlines()
			last = len(lines) - 1
			lines[last] = lines[last].replace('\r','').replace('\n','')
		with open(fname, 'w') as wr:
			wr.writelines(lines) 
	else:
		data1 = df.iloc[0:len(df)-1]
		data2 = df.iloc[[len(df)-1]]
		data1.to_csv(fname, sep='\t', encoding='utf-8', index = False)
		data2.to_csv(fname, sep='\t', encoding='utf-8', header= False, index = False, mode='a',line_terminator="")
		
def read_annotation_block(block, fname, header, chanindx):
	assert(block>=0)
	data = []
	data_overwrite = []
	with open(fname, 'rb') as fid:
		assert(fid.tell() == 0)
		blocksize = np.sum(header[1]['n_samps']) * header[0]['data_size']
		fid.seek(np.int64(header[0]['data_offset']) + np.int64(block) * np.int64(blocksize))
		read_idx = 0
		for i in range(header[0]['nchan']):
			read_idx += np.int64(header[1]['n_samps'][i]*header[0]['data_size'])
			buf = fid.read(np.int64(header[1]['n_samps'][i]*header[0]['data_size']))
			if i == chanindx:
				raw = buf.decode('latin-1', 'ignore')
				if re.search('[a-zA-Z]', raw):
					raw = [x for x in re.split(r'[^\x20-\x7e]', raw) if len(x) >0]
					if len(raw) > 3:
						raw = [raw[0], raw[1], ' '.join(raw[2:])]
					if any(x in raw[2] for x in {'Montage'}):
						 fid_loc = [(np.int64(header[0]['data_offset']) + np.int64(block) * np.int64(blocksize)) + read_idx, np.int64(header[1]['n_samps'][i]*header[0]['data_size'])]
						 data_overwrite.append(fid_loc + raw)
					else:
						 data.append(raw)
						 
	return data, data_overwrite

def read_annotation_samples(chanindx, fname, begsample, endsample, header):
	n_samps = header[1]['n_samps'][chanindx]
	begblock = int(np.floor((begsample) / n_samps))
	endblock = int(np.floor((endsample) / n_samps))
	data, overwrite = read_annotation_block(begblock, fname, header, chanindx)
	update_cnt = int(endblock/10)
	for block in range(begblock+1, endblock+1):
		data_temp, overwrite_temp = read_annotation_block(block, fname, header, chanindx)
		if data_temp:
			data.append(data_temp[0])
		if overwrite_temp:
			overwrite.append(overwrite_temp[0])
		if block == update_cnt:
			print('{}%'.format(int(np.ceil((update_cnt/endblock)*100))))
			update_cnt += int(endblock/10)
	
	if overwrite:
		for iblock in range(len(overwrite)):
			data.append([overwrite[iblock][2], overwrite[iblock][3], 'Montage Event'])
			with open(fname, 'r+b') as fid:
				assert(fid.tell() == 0)
				fid.seek(overwrite[iblock][0] - overwrite[iblock][1])
				times = overwrite[iblock][2] + '\x14\x14\x00' + overwrite[iblock][3] + '\x14' + 'Montage Event' +'\x14'
				ending = (times + ('\x00' * (overwrite[iblock][1]-len(times)))).encode('utf-8')
				assert(len(ending)==overwrite[iblock][1])
				fid.write(ending) # subject id
		
	df_data = pd.DataFrame({})
	for iannot in range(len(data)):
		ms = str(int((datetime.timedelta(seconds=float(data[iannot][0])).microseconds)/1000))
		data_temp = {'onset': data[iannot][0],
			         'duration': data[iannot][1],
			         'seconds': float(data[iannot][0]),
			         'time': str(datetime.timedelta(seconds=float(data[iannot][0]))).split('.')[0]+ms,
				     'event': data[iannot][2]}
					
		df_data = pd.concat([df_data, pd.DataFrame([data_temp])], axis = 0)
		
	df_data = df_data[['onset','duration','seconds','time','event']]
	df_data = df_data.sort_values(by='seconds').reset_index(drop=True)
	annot_fname = os.path.join(raw_file_path, ifile).split('.')[0] + '_annotations.tsv'
	_write_tsv(annot_fname, df_data)
	
	return df_data

def read_annotations(fname):
	file_in = EDFReader()
	file_in.open(os.path.join(fname))
	header = file_in.readHeader()
	file_in.close()
	chanindx = [i for i,x in enumerate(header[1]['ch_names']) if x.endswith('Annotations')][0]

	begsample = 0
	endsample = header[1]['n_samps'][chanindx] * header[0]['n_records'] - 1
	annotation_data = read_annotation_samples(chanindx, fname, begsample, endsample, header)
	
	annot_fname = os.path.join(raw_file_path, ifile).split('.')[0] + '_annotations.tsv'
	_write_tsv(annot_fname, annotation_data)

	return annotation_data

raw_file_path = r'H:\EDF_Studies\sub-021'
file_list = [x for x in os.listdir(raw_file_path) if x.lower().endswith('.edf')]

ifile = file_list[0]
fname = os.path.join(raw_file_path, ifile)

data_fname = os.path.join(raw_file_path, ifile)

fname = r'C:/Users/Greydon/Downloads/Input/EPL31_TST_0002/EPL31_TST_0002_01_SE01_IEEG_FULL/LastName~ FirstName_98a7b61a-5e8a7-a8fbc8de-72ba73ba8c8e.edf'
fname = r'/home/ggilmore/Downloads/graham/projects/rrg-akhanf/cfmm-bids/Khan/epi_iEEG/source_data/sub-015/from_suzan/X~ X_685e1963-4e5a-4827-8854-34b234aa8481.EDF'
read_annotations(fname)

fname
		   
import pyedflib

f = pyedflib.EdfReader(r'C:/Users/greydon/Downloads/input/sub-024/X~ X_0f659043-9dc3-4d81-8b70-6a194c9b0209.EDF')
annots = f.readAnnotations()
f.__del__()

deidentified = file_data[7]
fname = os.path.join(raw_file_path, deidentified['FileName'])
file_in = EDFReader()
file_in.open(os.path.join(fname))
deidentified_header = file_in.readHeader()
file_in.close()

deidentified_header[0]['birthdate']
deidentified_header[0]['gender']
deidentified_header[0]['recording_id']
deidentified_header[0]['subject_id']

def deidentify_edf(fname, isub):
	file_in = EDFReader()
	file_in.open(os.path.join(fname))
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
			fid.seek(80)
		
		if recording_id:
			fid.write(padtrim(recording_id, 80).encode('utf-8'))
		else:
			fid.seek(80)
			
	if '~' in fname.split(os.path.sep)[-1]:
		new_name = '_'.join([isub] + fname.split(os.path.sep)[-1].split('_')[1:]).split('.')[0] + '.edf'
		os.rename(fname, os.path.join(os.path.dirname(fname), new_name))
		
		
		f = pyedflib.EdfReader(fname)
		annots = f.readAnnotations()
		check_strings = ['Montage']
		if firstname:
			check_strings = check_strings + [firstname]
		if lastname:
			check_strings = check_strings + [lastname]
		
		for iannot in annots:
			if 'Montage' in iannot:
				if not iannot == 'Montage Event':
			if firstname:
				if firstname in iannot:
			if lastname:
				if lastname in iannot:
			
				
					
		chanindx = [i for i,x in enumerate(header[1]['ch_names']) if x.endswith('Annotations')][0]

		begsample = 0
		endsample = header[1]['n_samps'][chanindx] * header[0]['n_records'] - 1
		n_samps = header[1]['n_samps'][chanindx]
		begblock = int(np.floor((begsample) / n_samps))
		endblock = int(np.floor((endsample) / n_samps))
		data, overwrite = read_annotation_block(begblock, fname, header, chanindx)
		update_cnt = int(endblock/10)
		for block in range(begblock+1, endblock+1):
			data_temp, overwrite_temp = read_annotation_block(block, fname, header, chanindx)
			if data_temp:
				data.append(data_temp[0])
			if overwrite_temp:
				overwrite.append(overwrite_temp[0])
			if block == update_cnt:
				print('{}%'.format(int(np.ceil((update_cnt/endblock)*100))))
				update_cnt += int(endblock/10)
		
		if overwrite:
			for iblock in range(len(overwrite)):
				data.append([overwrite[iblock][2], overwrite[iblock][3], 'Montage Event'])
				with open(fname, 'r+b') as fid:
					assert(fid.tell() == 0)
					fid.seek(overwrite[iblock][0] - overwrite[iblock][1])
					times = overwrite[iblock][2] + '\x14\x14\x00' + overwrite[iblock][3] + '\x14' + 'Montage Event' +'\x14'
					ending = (times + ('\x00' * (overwrite[iblock][1]-len(times)))).encode('utf-8')
					assert(len(ending)==overwrite[iblock][1])
					fid.write(ending) # subject id
			
		df_data = pd.DataFrame({})
		for iannot in range(len(data)):
			ms = str(int((datetime.timedelta(seconds=float(data[iannot][0])).microseconds)/1000))
			data_temp = {'onset': data[iannot][0],
				         'duration': data[iannot][1],
				         'seconds': float(data[iannot][0]),
				         'time': str(datetime.timedelta(seconds=float(data[iannot][0]))).split('.')[0]+ms,
					     'event': data[iannot][2]}
						
			df_data = pd.concat([df_data, pd.DataFrame([data_temp])], axis = 0)
			
		df_data = df_data[['onset','duration','seconds','time','event']]
		df_data = df_data.sort_values(by='seconds').reset_index(drop=True)
		annot_fname = os.path.join(raw_file_path, ifile).split('.')[0] + '_annotations.tsv'
		_write_tsv(annot_fname, df_data)
	
			

raw_file_path = r'H:\EDF_Studies\sub-021'
file_list = [x for x in os.listdir(raw_file_path) if x.lower().endswith('.edf')]

for ifile in range(len(file_list)):
	fname = os.path.join(raw_file_path, file_list[ifile])
	with open(fname, 'r+b') as fid:
		assert(fid.tell() == 0)
		fid.seek(8)
		fid.write(padtrim('X X X ' + raw_file_path.split(os.path.sep)[-1], 80).encode('utf-8'))
		fid.write(padtrim('Startdate X X X X', 80).encode('utf-8'))
		
	new_name = '_'.join([raw_file_path.split(os.path.sep)[-1]] + file_list[ifile].split('_')[1:])
	os.rename(fname, os.path.join(raw_file_path, new_name))



