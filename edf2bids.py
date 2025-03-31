
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 13:49:51 2018

@author: Greydon
"""
import os
import numpy as np
import pandas as pd
#pd.set_option('precision', 6)
import datetime
import shutil
from PySide6 import QtCore
import time
import re
import io
import traceback, sys
import gzip
from datetime import timedelta
import pdb
from edf_embedded_annotations_redactor import anonymize_edf_complete


from ext_lib.edflibpy import EDFreader as EDFLIBReader
from helpers import EDFReader, bidsHelper, fix_sessions, sec2time, deidentify_edf


DemoMode = False

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
    errorEvent = QtCore.Signal(tuple)
    
class edf2bids(QtCore.QRunnable):
    """
    This class is a thread, which manages one thread of control within the GUI.
    
    :param new_sessions: Dictionary containing information about each subject. Wether there are new sessions to process in the input directory. 
    :type new_sessions: dictionary
    :param file_info: Information about each subject file in the input directory
    :type file_info: dictionary
    :param input_path: Path to the input directory.
    :type input_path: string
    :param output_path: path to the output directory
    :type output_path: string
    :param coordinates: Optional list of electrode coordinates (x,y,z)
    :type coordinates: list or None
    :param make_dir: Make the directory
    :type make_dir: boolean
    :param overwrite: If duplicate data is present in the output directory overwrite it.
    :type overwrite: boolean
    :param verbose: Print out process steps.
    :type verbose: boolean
    :param annotation_extract: Extract annotations from the edf file.
    :type annotation_extract: boolean
    :param compress: Compress the edf file
    :type compress: boolean
    
    """
    
    def __init__(self):    
        super(edf2bids, self).__init__()
        
        self.new_sessions = {}
        self.file_info = []
        self.chan_label_file = []
        self.input_path = []
        self.output_path = []
        self.script_path = []
        self.coordinates = []
        self.make_dir = []
        self.overwrite = []
        self.verbose = []
        self.deidentify_source = []
        self.gzip_edf = False
        self.offset_date = []
        self.bids_settings = {}
        self.dry_run = []
        
        self.signals = WorkerSignals()
        
        self.running = False
        self.userAbort = False
        self.is_paused = False
        self.is_killed = False
        self.isError=False
        
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
        
    def write_annotations(self, source_fname, data_fname, callback, deidentify=True):
        self._annotations_data(source_fname, data_fname, callback, deidentify)
    
    def overwrite_annotations(self, events, data_fname, identity_idx, tal_indx, strings, action):
        
        
        for ident in identity_idx:
            block_chk = events[ident][-1]
            assert(block_chk>=0)
            
            if (data_fname.lower().endswith(".edf")):
                fid=open(data_fname, "rb")
            elif (data_fname.lower().endswith(".edfz")) or (data_fname.lower().endswith(".edf.gz")):
                fid=gzip.open(data_fname, "rb")

            assert(fid.tell() == 0)
            fid.seek((block_chk-(self.header['chan_info']['n_samps'][tal_indx]*2)), io.SEEK_SET)
            cnv_buf=bytearray(self.header['chan_info']['n_samps'][tal_indx]*2)
            fid.readinto(cnv_buf)
            fid.close()

            if isinstance(strings, dict):
                replace_idx = [i for i,x in enumerate(strings.keys()) if re.search(x, events[ident][2], re.IGNORECASE)]
            else:
                replace_idx = [i for i,x in enumerate(strings) if re.search(x, events[ident][2], re.IGNORECASE)]
            
            new_block=[]
            for irep in replace_idx:
                if action == 'replaceExact':
                    if new_block:
                        new_block = bytearray(re.sub(bytes(strings[irep],'latin-1'),bytes(''.join(np.repeat('X', len(strings[irep]))),'latin-1'),new_block,flags=re.I))
                        events[ident][2] = re.sub(strings[irep],''.join(np.repeat('X', len(strings[irep]))),events[ident][2],flags=re.I)
                    else:
                        new_block = bytearray(re.sub(bytes(strings[irep],'latin-1'),bytes(''.join(np.repeat('X', len(strings[irep]))),'latin-1'),cnv_buf,flags=re.I))
                        events[ident][2] = re.sub(strings[irep],''.join(np.repeat('X', len(strings[irep]))),events[ident][2],flags=re.I)
                    
                    assert(len(new_block)==len(cnv_buf))
                
                elif action == 'replaceMatch':
                    replace_string = list(strings.values())[irep]
                    new_block = cnv_buf.replace(bytes(events[ident][2],'latin-1'),bytes(replace_string,'latin-1'))
                    events[ident][2] = replace_string
                    new_block = new_block+bytes('\x00'*(len(cnv_buf)-len(new_block)),'latin-1')
                    
                    assert(len(new_block)==len(cnv_buf))
                    
            if new_block:
                

                assert(fid.tell() == 0)
                fid.seek((block_chk-(self.header['chan_info']['n_samps'][tal_indx]*2)), io.SEEK_SET)
                if fid.tell()==(block_chk-(self.header['chan_info']['n_samps'][tal_indx]*2)):
                    fid.write(new_block)
                fid.close()
        
        return events

    def _annotations_data(self, source_name, data_fname, callback, deidentify):
        """
        Constructs an annotations data tsv file about patient specific events from edf file.
        
        :param file_info_run: File header information for specific recording.
        :type file_info_run: dictionary
        :param annotation_fname: Filename for the annotations tsv file.
        :type annotation_fname: string
        :param data_fname: Path to the raw data file for specific recording.
        :type data_fname: string
        :param overwrite: If duplicate data is present in the output directory overwrite it.
        :type overwrite: boolean
        :param verbose: Print out process steps.
        :type verbose: boolean
        
        """        
        file_in = EDFReader()
        file_in.open(source_name)
        self.header = file_in.readHeader()
        
        overwrite_exact = [self.header['meas_info']['firstname'], self.header['meas_info']['lastname']]
#         overwrite_exact = [header['meas_info']['firstname'], header['meas_info']['lastname']]
        overwrite_exact = [x for x in overwrite_exact if x is not None]
        overwrite_match={
                        'montage': 'Montage Event'
                        }
        remove_strings = []
        # pdb.set_trace()  # Breakpoint inside thread

        tal_indx = [i for i,x in enumerate(self.header['chan_info']['ch_names']) if x.endswith('Annotations')][0]
#         tal_indx = [i for i,x in enumerate(header['chan_info']['ch_names']) if x.endswith('Annotations')][0]
        
        callback.emit('...')
        
        hdl = EDFLIBReader(source_name)
        events=hdl.annotationslist
        events=[list(x) for x in events]

        if deidentify:
            if overwrite_exact:
                ### Replace any identifier strings
                identity_idx = [i for i,x in enumerate(events) if any(re.search(substring, x[2], re.IGNORECASE) for substring in overwrite_exact) and not re.search('montage', x[2], re.IGNORECASE)]
                if identity_idx:
                    events = self.overwrite_annotations(events, data_fname, identity_idx, tal_indx, overwrite_exact, 'replaceExact')
            
            if overwrite_match:
                identity_idx = [i for i,x in enumerate(events) if any(re.search(substring, x[2], re.IGNORECASE) for substring in overwrite_match.keys()) and not any(re.match(substring, x[2], re.IGNORECASE) for substring in list(overwrite_match.values()))]
                if identity_idx:
                    events = self.overwrite_annotations(events, data_fname, identity_idx, tal_indx, overwrite_match, 'replaceMatch')
        
            if remove_strings:
                ### Remove unwanted annoations
                identity_idx = [i for i,x in enumerate(events) if any(re.search(substring, x[2], re.IGNORECASE) for substring in remove_strings)]
                if identity_idx:
                    events = self.overwrite_annotations(events, data_fname, identity_idx, tal_indx, remove_strings, 'remove')
        
        annotation_data = pd.DataFrame({})
        if events:
            fulldate = datetime.datetime.strptime(self.header['meas_info']['meas_date'], '%Y-%m-%d %H:%M:%S')
            for iannot in events:
                data_temp = {'onset': iannot[0]/10000000,
                             'duration': iannot[1],
                             'time_abs': (fulldate + datetime.timedelta(seconds=(iannot[0]/10000000)+float(self.header['meas_info']['millisecond']))).strftime('%H:%M:%S.%f'),
                             'time_rel': sec2time(iannot[0]/10000000, 6),
                             'event': iannot[2]}
                annotation_data = pd.concat([annotation_data, pd.DataFrame([data_temp])], axis = 0)
            
        annotation_data.to_csv(self.annotation_fname, sep='\t', index=False, na_rep='n/a', float_format='%.3f')
    
    def copyLargeFile(self, src, dest, callback=None, buffer_size=65536*10, max_retries=3, sub_log_dir = ""):
        # pdb.set_trace()  # Breakpoint inside thread
        
        
        success = anonymize_edf_complete(src, dest,log_dir = sub_log_dir)
        
        # try:
        #     if not os.path.exists(src):
        #         raise FileNotFoundError(f"The source file '{src}' does not exist.")
    
        #     total_size = os.path.getsize(src)
        #     update_interval = int(total_size / 1000)  # Update progress every 0.1%
        #     next_update = update_interval
        #     start_time = time.time()
            
        #     # Create single line for progress updates
        #     self.signals.progressEvent.emit(f"\n{self.conversionStatusText}")
    
        #     def write_progress(copied):
        #         elapsed_time = time.time() - start_time
        #         speed = copied / elapsed_time if elapsed_time > 0 else 0
        #         remaining_time = (total_size - copied) / speed if speed > 0 else 0
        #         estimated_completion = timedelta(seconds=int(remaining_time))
                
        #         # Use carriage return to update same line
        #         print(f"\rWorking on {os.path.basename(src)} => {os.path.basename(dest)} Progress: {(copied / total_size) * 100:5.2f}% - ETA: {estimated_completion}", 
        #               end='', flush=True)
    
        #     mode = "wb" if dest.lower().endswith(".edf") else "wb" if dest.lower().endswith((".edfz", ".edf.gz")) else None
        #     open_func = open if mode == "wb" else gzip.open
    
        #     with open_func(dest, mode) as fdest:
        #         with open(src, 'rb') as fsrc:
        #             copied = 0
        #             while True:
        #                 while self.is_paused:
        #                     time.sleep(0.1)
        #                 if self.is_killed:
        #                     self.running = False
        #                     raise WorkerKilledException
    
        #                 retry_count = 0
        #                 while retry_count < max_retries:
        #                     try:
        #                         buf = fsrc.read(buffer_size)
        #                         if not buf:
        #                             break
        #                         fdest.write(buf)
        #                         break
        #                     except OSError as e:
        #                         retry_count += 1
        #                         if retry_count >= max_retries:
        #                             raise OSError(f"Buffering error: {e}")
        #                         time.sleep(0.1)
    
        #                 if DemoMode:
        #                     break
        #                 if not buf:
        #                     break
    
        #                 copied += len(buf)
        #                 if copied >= next_update:
        #                     write_progress(copied)
        #                     next_update += update_interval
    
        #     # Clear progress line and print completion
        #     # print("\r" + " " * 80 + "\r", end='', flush=True)  # Clear the line
        #     print(f"\r\nFile copy completed - {total_size:,} bytes transferred", flush=True)
    
        #     if callback is not None:
        #         callback.emit('100%')
    
        # except Exception as e:
        #     print(f"\nError during file copy: {str(e)}")
        #     raise


#%%
#     isub = list(new_sessions)[0]
#     new_sessions[isub]['session_labels']=sum(new_sessions[isub]['session_labels'],[])
#     new_sessions[isub]['all_sessions']=sum(new_sessions[isub]['all_sessions'],[])
#     new_sessions[isub]['num_sessions']=len(np.unique(new_sessions[isub]['session_labels']))
#    values = new_sessions[isub]
#     edf2b=edf2bids()
    @QtCore.Slot()
    def run(self):
        """
        Main loop for building BIDS database.
                
        """
        if not self.userAbort:
            self.running = True
        
        participants_fname = bidsHelper(output_path=self.output_path).write_participants(return_fname=True)
        if os.path.exists(participants_fname):
            self.participant_tsv = pd.read_csv(participants_fname, sep='\t')
            
        try:
            for isub in list(self.file_info):
                if self.file_info[isub]:
                    subject_dir = self.file_info[isub][0]['SubDir']
                    raw_file_path = os.path.join(self.input_path, subject_dir)
    #                 subject_dir = file_info[isub][0][0]['SubDir']
    #                 raw_file_path = os.path.join(input_path, subject_dir)
                    
                    values = self.new_sessions[isub]
                        
                    if values['newSessions']:
                        sessions_fix = [x for x in values['session_changes'] if x[0] != x[1]]
                        if sessions_fix:
                            fix_sessions(sessions_fix, values['num_sessions'], self.output_path, isub)
                        
                        update_info=[]
                        for idx, isession in enumerate(np.unique(values['session_labels'])):
                            update_info.append([self.file_info[isub][i] for i,x in enumerate(values['session_labels']) if x == isession])
                            
                        self.file_info[isub]=update_info
                        
                        for ises in range(len(self.file_info[isub])):
                            while self.is_paused:
                                time.sleep(0)
                            
                            if self.is_killed:
                                self.running = False
                                raise WorkerKilledException
                        
                            file_data = self.file_info[isub][ises]
                            session_id = np.unique(values['session_labels'])[ises].split('-')[-1]
                            
                            self.conversionStatusText = '\nStarting conversion: session {} of {} for {} at {}'.format(str(ises+1), str(len(np.unique(values['session_labels']))), isub, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                            self.signals.progressEvent.emit(self.conversionStatusText)
                        
                            num_runs = len(file_data)
                            
                            for irun in range(num_runs):
                                while self.is_paused:
                                    time.sleep(0)
                                
                                if self.is_killed:
                                    self.running = False
                                    raise WorkerKilledException
                                
                                if 'Scalp' in file_data[irun]['RecordingType']:
                                    kind = 'eeg'
                                elif 'iEEG' in file_data[irun]['RecordingType']:
                                    kind = 'ieeg'
                                
                                task_id = file_data[irun]['RecordingLength']
                                
                                if 'Ret' in file_data[irun]['Retro_Pro']:
                                    task_id = task_id + 'ret'
                                
                                run_num = str(irun+1).zfill(2)
                                
                                bids_helper=bidsHelper(subject_id=isub, session_id=session_id, kind=kind, task_id=task_id, run_num=run_num, output_path=self.output_path, bids_settings=self.bids_settings, make_sub_dir=True)
    #                             bids_helper = bidsHelper(subject_id=isub, session_id=session_id, kind=kind, task_id=task_id, run_num=run_num, output_path=output_path, bids_settings=bids_settings, make_sub_dir=True)
                                
                                data_fname = bids_helper.make_bids_filename(suffix = kind + '.edf')
                                source_name = os.path.join(raw_file_path, file_data[irun]['FileName'])
                                self.annotation_fname = bids_helper.make_bids_filename(suffix='events.tsv')
                                if not self.dry_run and DemoMode == False:
                                    if self.deidentify_source:
                                        self.write_annotations(source_name, data_fname, self.signals.progressEvent, deidentify=True)
                                        source_name, epochLength = deidentify_edf(source_name, data_fname, isub, self.offset_date, True)
                                        self.bids_settings['json_metadata']['EpochLength'] = epochLength
    #                                     edf2b.copyLargeFile(source_name, data_fname)
                                    
                                    self.copyLargeFile(source_name, data_fname, self.signals.progressEvent, sub_log_dir = subject_dir)
                                    
                                    if not self.deidentify_source:
                                        self.write_annotations(source_name, data_fname, self.signals.progressEvent, deidentify=False)
                                        temp_name, epochLength = deidentify_edf(data_fname,data_fname, isub, self.offset_date, False)
                                        self.bids_settings['json_metadata']['EpochLength'] = epochLength
                                    
                                    
                                    if file_data[irun]['chan_label']:
                                        file_in = EDFReader()
                                        file_in.open(data_fname)
                                        chan_label_file=file_in.chnames_update(os.path.join(raw_file_path, file_data[irun]['chan_label'][0]), self.bids_settings, write=True)
                                    elif file_data[irun]['ses_chan_label']:
                                        file_in = EDFReader()
                                        file_in.open(data_fname)
                                        chan_label_file=file_in.chnames_update(os.path.join(raw_file_path, file_data[irun]['ses_chan_label'][0]), self.bids_settings, write=True)
                                    
                                else:
                                    self.write_annotations(source_name, source_name, self.signals.progressEvent, deidentify=False)
                                    if self.deidentify_source:
                                        source_name = os.path.join(raw_file_path, file_data[irun]['FileName'])
                                        source_name, epochLength = deidentify_edf(source_name, data_fname, isub, self.offset_date, True)
                                        self.bids_settings['json_metadata']['EpochLength'] = epochLength
                                    else:
                                        self.bids_settings['json_metadata']['EpochLength'] = 0
                                
                                scan_fname=data_fname.split(isub+os.path.sep)[-1].replace(os.path.sep,'/')
                                if self.gzip_edf:
                                    scan_fname=scan_fname+'.gz'
                                
                                bids_helper.write_scans(scan_fname, file_data[irun], self.offset_date)
                                
                                bids_helper.write_channels(file_data[irun])
                                bids_helper.write_sidecar(file_data[irun])

                                if self.gzip_edf and not self.dry_run and not DemoMode:
                                    data_fname_old=data_fname
                                    data_fname=os.path.splitext(data_fname)[0]+'.edf.gz'
                                    with open(data_fname_old, 'rb') as f_in:
                                        with gzip.open(data_fname, 'wb') as f_out:
                                            shutil.copyfileobj(f_in, f_out)
                                    os.remove(data_fname_old)
                            
                            bids_helper.write_electrodes(file_data[0], coordinates=None)
                            
                            code_output_path = os.path.join(self.output_path, 'code', 'edf2bids')
                            code_path = bids_helper.make_bids_folders(path_override=code_output_path, make_dir=True)
                            
                            shutil.copy(os.path.join(self.script_path, 'edf2bids.py'), code_path)
                            shutil.copy(os.path.join(self.script_path, 'helpers.py'), code_path)
                        
                            self.conversionStatusText = 'Finished conversion: session {} of {} for {} at {}'.format(str(ises+1), str(len(np.unique(values['session_labels']))), isub, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                            self.signals.progressEvent.emit(self.conversionStatusText)
                        
                        time.sleep(0.1)
                        
                        if isub not in list(self.participant_tsv['participant_id']):
                            bids_helper.write_participants(self.file_info[isub][0])
                            self.participant_tsv = pd.read_csv(participants_fname, sep='\t')
                    else:
                        self.conversionStatusText = 'Participant {} already exists in the dataset! \n'.format(isub)
                        self.signals.progressEvent.emit(self.conversionStatusText)
                
        except:
            if not self.is_killed and not self.is_paused:
                self.running = False
                self.isError=True
                
                exctype, value = sys.exc_info()[:2]
                self.signals.errorEvent.emit((exctype, value, traceback.format_exc()))
                print(exctype, value)

                pass
            else:
                raise WorkerKilledException
        
        finally:
            self.running = False
            if not self.isError:
                self.signals.finished.emit()





