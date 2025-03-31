#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:06:23 2020

@author: greydon
"""
import os
import pandas as pd
#pd.set_option('precision', 6)
import shutil
from PySide6 import QtCore
import time
import re
import datetime
import numpy as np
import glob
import zipfile

class WorkerKilledException(Exception):
	pass

class WorkerSignals(QtCore.QObject):
	'''
	Defines the signals available from a running worker thread.

	Supported signals are:

	finished
		No data
	
	error
		`tuple` (exctype, value, traceback.format_exc() )
	
	result
		`object` data returned from processing, anything

	progress
		`int` indicating % progress 

	'''
	finished = QtCore.Signal()
	progressEvent = QtCore.Signal(str)
	
class bids2spred(QtCore.QRunnable):
		
	def __init__(self):	
# 		QtCore.QThread.__init__(self)
		super(bids2spred, self).__init__()
		
		self.output_path = []
		self.script_path = []
		
		self.signals = WorkerSignals()
		
		self.running = False
		self.userAbort = False
		self.is_paused=False
		self.is_killed = False
		
	def stop(self):
		self.running = False
		self.userAbort = True
	
	def pause(self):
		if not self.is_paused:
			self.is_paused=True
		else:
			self.is_paused=False
	
	def kill(self):
		self.is_killed = True
	
# 	output_path = r'/media/veracrypt6/projects/eplink/walkthrough_example/working_dir/output/bids_old'
	@QtCore.Slot()
	def run(self):
		"""
		Main loop.
				
		"""
		if not self.userAbort:
			self.running = True
		
		try:
			self.conversionStatusText = 'Converting directory to SPReD format...\n'
			self.signals.progressEvent.emit(self.conversionStatusText)
			
			folders = [x for x in os.listdir(self.output_path) if os.path.isdir(os.path.join(self.output_path, x)) and 'code' not in x]
# 			folders = [x for x in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, x)) and 'code' not in x]
			output_folders = ['_'.join([''.join(' '.join(re.split('(\d+)', x.split('-')[-1])).split()[:2])] + ' '.join(re.split('(\d+)', x.split('-')[-1])).split()[2:]) for x in folders]
			sub_cnt = 0
			for isub in folders:
				while self.is_paused:
					time.sleep(0)
				
				if self.is_killed:
					self.running = False
					raise WorkerKilledException
					
				ses_folders = [x for x in os.listdir(os.path.sep.join([self.output_path, isub])) if os.path.isdir(os.path.sep.join([self.output_path, isub, x]))]
# 				ses_folders = [x for x in os.listdir(os.path.sep.join([output_path, isub])) if os.path.isdir(os.path.sep.join([output_path, isub, x]))]
				
				session_split = [[x for x in i if x.isdigit()] for i in ses_folders]
				ses_folders_output = ['_'.join([x[0]+x[1],'SE'+x[2]+x[3], 'EEG']) for x in session_split]
				ses_cnt = 0
				for ises in ses_folders:
					self.conversionStatusText = 'Performing SPReD conversion: session {} of {} for {} at {}'.format(str(ses_cnt+1), str(len(ses_folders)), isub, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
					self.signals.progressEvent.emit(self.conversionStatusText)
					
					old_subfold = [x for x in os.listdir(os.path.sep.join([self.output_path, isub, ises])) if os.path.isdir(os.path.sep.join([self.output_path, isub, ises, x]))]
					old_subfold = [x for x in os.listdir(os.path.sep.join([self.output_path, isub, ises])) if os.path.isdir(os.path.sep.join([self.output_path, isub, ises, x]))]
					old_edf_name = [x.split('task-')[1].split('_')[0] for x in os.listdir(os.path.sep.join([self.output_path, isub, ises, old_subfold[0]])) if x.endswith('.json')]
					
# 					old_subfold = [x for x in os.listdir(os.path.sep.join([output_path, isub, ises])) if os.path.isdir(os.path.sep.join([output_path, isub, ises, x]))]
# 					old_subfold = [x for x in os.listdir(os.path.sep.join([output_path, isub, ises])) if os.path.isdir(os.path.sep.join([output_path, isub, ises, x]))]
# 					old_edf_name = [x.split('task-')[1].split('_')[0] for x in os.listdir(os.path.sep.join([output_path, isub, ises, old_subfold[0]])) if x.endswith('.json')]

					for itask in np.unique(old_edf_name):
						
						while self.is_paused:
							time.sleep(0)
						
						if self.is_killed:
							self.running = False
							raise WorkerKilledException
						
						if 'clip' in itask:
							suffix = '_CLIP'
						elif 'full' in itask:
							suffix = '_FULL'
						elif 'stim' in itask:
							suffix = '_STIM'
						
						if any(x == itask for x in {'full', 'clip','stim'}):
							suffix += '_PRO'
						elif any(x == itask for x in {'fullret', 'clipret','stimret'}):
							suffix += '_RET'
						
						new_subfolder = '_'.join([output_folders[sub_cnt], ses_folders_output[ses_cnt]+'_'+ old_subfold[0].upper() + suffix])
						old_fold = os.path.sep.join([self.output_path, isub, ises, old_subfold[0]])
						new_fold = os.path.sep.join([self.output_path, 'SPReD', output_folders[sub_cnt], new_subfolder, new_subfolder])
						
						#old_fold = os.path.sep.join([output_path, isub, ises, old_subfold[0]])
						#new_fold = os.path.sep.join([output_path, 'SPReD', output_folders[sub_cnt], new_subfolder, new_subfolder])
	
						if not os.path.exists(new_fold):
							os.makedirs(new_fold)
						
						
						electrodes_input = [x for x in os.listdir(old_fold) if x.endswith('electrodes.tsv')]
						if not os.path.exists(os.path.join(new_fold, electrodes_input[0])):
							shutil.copy2(os.path.join(old_fold, electrodes_input[0]), os.path.join(new_fold, electrodes_input[0]))
						
						for filePath in glob.glob(old_fold+f'/*task-{itask}*'):
							shutil.move(filePath, os.path.join(new_fold, os.path.basename(filePath)))
						
# 						shutil.make_archive( (os.path.dirname(new_fold), 'zip', os.path.dirname(new_fold))
						with zipfile.ZipFile(os.path.dirname(new_fold)+'.zip', 'w', zipfile.ZIP_DEFLATED) as ziph:
							for root, dirs, files in os.walk(os.path.dirname(new_fold)):
								for file in files:
									ziph.write(os.path.join(root, file), os.path.join(os.path.basename(new_fold), file))
						
						shutil.rmtree(os.path.dirname(new_fold))
						
					if ises == ses_folders[-1]:
						self.conversionStatusText = ''
						self.signals.progressEvent.emit(self.conversionStatusText)
					
					ses_cnt += 1
				
				sub_cnt += 1
				
				time.sleep(0.1)
			
			folders = [x for x in os.listdir(self.output_path) if 'SPReD' not in x]
			
			if not os.path.exists(os.path.sep.join([self.output_path, 'bids_old'])):
				os.makedirs(os.path.sep.join([self.output_path, 'bids_old']))
								
			for ifiles in folders:
				old_file = os.path.sep.join([self.output_path, ifiles])
				new_file = os.path.sep.join([self.output_path, 'bids_old', ifiles])
				shutil.move(old_file, new_file)
			
		except WorkerKilledException:
			self.running = False
			pass
		
		finally:
			self.running = False
			self.signals.finished.emit()
			
				