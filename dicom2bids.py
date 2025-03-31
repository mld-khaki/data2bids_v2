#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:06:23 2020

@author: greydon
"""
import os
import pandas as pd
#pd.set_option('precision', 6)
from PySide6 import QtCore
import pydicom
from dicognito.anonymizer import Anonymizer
import datetime
import shutil
import tarfile
import zipfile
import uuid
import time

class WorkerKilledException(Exception):
	pass


class WorkerSignals(QtCore.QObject):
	'''
	Defines the signals available from a running worker thread.

	Supported signals are:

	finished
		No data
	
	progressEvent
		`str`

	'''
	finished=QtCore.Signal()
	progressEvent=QtCore.Signal(str)


class dicom2bids(QtCore.QRunnable):
	
	def __init__(self):	
		super(dicom2bids, self).__init__()
		
		self.input_path=[]
		self.output_path=[]
		self.imaging_data=[]
		
		self.signals=WorkerSignals()
		
		self.running=False
		self.userAbort=False
		self.is_paused=False
		self.is_killed=False
		self.compressed_exts=('.tar', '.tgz', '.tar.gz', '.tar.bz2', '.zip')
	
	def stop(self):
		self.running=False
		self.userAbort=True
	
	def pause(self):
		if not self.is_paused:
			self.is_paused=True
		else:
			self.is_paused=False
	
	def kill(self):
		self.is_killed=True
	
	def _extract(self, filename, to_dir):
		'''
		extract compressed files
		'''
		if (filename.endswith(".tar")):
			c_file=tarfile.open(filename, "r:")
		elif (filename.endswith(".tar.gz")) or (filename.endswith(".tgz")):
			c_file=tarfile.open(filename, "r:gz")
		elif (filename.endswith(".tar.bz2")):
			c_file=tarfile.open(filename, "r:bz2")
		elif (filename.endswith(".zip")):
			c_file=zipfile.ZipFile(filename)
		
		c_file.extractall(to_dir)
		c_file.close()
		
		dicom_files=[]
		for root, directories, filenames in os.walk(to_dir):
			for filename in filenames:
				full_filename=os.path.join(root, filename)
				dicom_files.append(full_filename)
		
		return dicom_files
	
	@QtCore.Slot()
	def run(self):
		"""
		Main loop.
		
		"""
		if not self.userAbort:
			self.running=True
		
		try:
			self.conversionStatusText='De-identifying imaging data...'
			self.signals.progressEvent.emit(self.conversionStatusText)
			#isub=list(imaging_data.keys())[0]
			print(list(self.imaging_data.keys()))
			for isub in list(self.imaging_data.keys()):
				
				if self.imaging_data[isub]['imaging_dir']:
					self.conversionStatusText='\nDe-identifying imaging data for subject {} at {}'.format(isub, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
					self.signals.progressEvent.emit(self.conversionStatusText)
					
					while self.is_paused:
						time.sleep(0)
					
					if self.is_killed:
						self.running=False
						raise WorkerKilledException
					
					outdir=os.path.join(self.output_path, 'imaging', self.imaging_data[isub]['orig_sub_dir'])
					base=os.path.join(self.input_path, self.imaging_data[isub]['orig_sub_dir'], self.imaging_data[isub]['imaging_dir'][0])
					
					#outdir=os.path.join(output_path, 'imaging', imaging_data[isub]['orig_sub_dir'])
					#base=os.path.join(input_path, imaging_data[isub]['orig_sub_dir'], imaging_data[isub]['imaging_dir'][0])
					
					study_folders=[x for x in os.listdir(base) if os.path.isdir(os.path.join(base, x))]
					
					anonymizer=Anonymizer()
					for istudy in study_folders:
						
						dicom_files=[]
						remove_dir=[]
						
						for root, folders, files in os.walk(os.path.join(base, istudy)):
							for file in files:
								
								while self.is_paused:
									time.sleep(0)
								
								if self.is_killed:
									self.running=False
									raise WorkerKilledException
								
								fullpath=os.path.abspath(os.path.join(root,file))
								if fullpath.endswith(self.compressed_exts):
									file_ext=[x for x in self.compressed_exts if fullpath.endswith(x)]
									
									if len(file_ext)==1:
										output_dir=os.path.join(outdir, 'tmp', os.path.basename(fullpath).split(file_ext[0])[0].replace('.','_') + '_' + str(uuid.uuid4()))
										
										if not os.path.exists(output_dir):
											os.makedirs(output_dir)
										
										remove_dir.append(output_dir)
										dicom_files=dicom_files + self._extract(fullpath, output_dir)
								else:
									dicom_files.append(fullpath)
						
						for original_filename in dicom_files:
							
							session_name='_'.join(istudy.split('_')[:-1] + ['SE' + istudy.split('_')[-1], 'MRI'])
							
							outname=os.path.sep.join([outdir, session_name, original_filename.split(f'{os.sep}tmp{os.sep}')[-1].split(istudy)[-1][len(os.sep):]])
							
							while self.is_paused:
								time.sleep(0)
							
							if self.is_killed:
								self.running=False
								raise WorkerKilledException
							
							if not os.path.exists(os.path.dirname(outname)):
								os.makedirs(os.path.dirname(outname))
							
							with pydicom.dcmread(original_filename) as dataset:
								anonymizer.anonymize(dataset)
								
								if 'PatientID' not in dataset:
									dataset.add_new('0x00100010', 'PN', session_name)
								else:
									dataset.PatientName=session_name
								
								dataset.save_as(outname)
						
						if remove_dir:
							shutil.rmtree(os.path.join(outdir, 'tmp'))
						
						out_zip=os.path.sep.join([outdir, session_name])
						shutil.make_archive(out_zip, 'zip', os.path.dirname(out_zip), os.path.basename(out_zip))
						shutil.rmtree(out_zip)
						
						self.conversionStatusText=f"Finished {isub} session {istudy} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
						self.signals.progressEvent.emit(self.conversionStatusText)
		
		except WorkerKilledException:
			self.running=False
			pass
		
		finally:
			self.running=False
			self.signals.finished.emit()
