# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:14:21 2019

@author: Greydon
"""
import os
import numpy as np
import pandas as pd
#pd.set_option('precision', 6)
import json
import datetime
import warnings
import re
from struct import unpack
from math import floor
import shutil
import glob
from collections import OrderedDict
import sys
import struct
# from PySide6 import QtWidgets
#from PyQt5 import QtWidgets
import gzip
import traceback


##############################################################################
#                                 HELPERS                                    #
##############################################################################
def padtrim(buf, num):
    """
    Adds padding to string data.
    
    :param buf: input string to pad
    :type buf: string
    :param num: Length of desired output buffer.
    :type num: integer

    :return buffer: The input string padded to desired size.
    :type butter: string

    """
    num -= len(buf)
    if num>=0:
        # pad the input to the specified length
        buffer = (str(buf) + ' ' * num)
    else:
        # trim the input to the specified length
        buffer = (buf[0:num])
    
    return bytes(buffer, 'latin-1')

def sorted_nicely(lst):
    """
    Sorts the given iterable in the way that is expected.
    
    :param lst: The iterable to be sorted.
    :type lst: list

    :return sorted_lst: The input string padded to desired size.
    :type sorted_lst: list

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    sorted_lst = sorted(lst, key = alphanum_key)
    
    return sorted_lst

def partition(iterable):
    """
    Seperates list of strings into alpha and digit.
    
    :param iterable: The iterable to seperated.
    :type iterable: list

    :return values: List of lists containing seperated input strings.
    :type values: list

    """
    values = []
    for item in iterable:
        if len(re.findall(r"([a-zA-Z]+)([0-9]+)",item))>1:
            first = "".join(list(re.findall(r"([a-zA-Z]+)([0-9]+)([a-zA-Z]+)",item)[0]))
            second = list(re.findall(r"([a-zA-Z]+)([0-9]+)",item)[-1])[-1]
            values.append([first, second])
        elif list(re.findall(r"([a-zA-Z]+)([0-9]+)",item)):
            values.append(list(re.findall(r"([a-zA-Z]+)([0-9]+)",item)[0]))
        else:
            values.append(["".join(x for x in item if not x.isdigit()), "".join(sorted(x for x in item if x.isdigit()))])
    
    return values

def determine_groups(iterable):
    """
    Identifies unique strings in a list of strings which include alphas.
    
    :param iterable: The iterable to seperated into unique groups.
    :type iterable: list

    :return values: List of lists ccontaining seperated string groups.
    :type values: list

    """
    values = []
    for item in iterable:
        if len(re.findall(r"([a-zA-Z]+)([0-9]+)",item))>1:
            first = "".join(list(re.findall(r"([a-zA-Z]+)([0-9]+)([a-zA-Z]+)",item)[0]))
            values.append(first)
        else:
            values.append("".join(x for x in item if not x.isdigit()))
    
    values = list(set(values))
    
    return values

def sec2time(sec, n_msec=3):
    ''' Convert seconds to 'D days, HH:MM:SS.FFF' '''
    if hasattr(sec,'__len__'):
        return [sec2time(s) for s in sec]
    neg=False
    if sec <0:
        neg=True
        sec = sec*-1
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if n_msec > 0:
        pattern = '%%02d:%%02d:%%0%d.%df' % (n_msec+3, n_msec)
    else:
        pattern = r'%02d:%02d:%02d'
    if d == 0:
        if neg:
            return '-' + pattern % (h, m, s)
        else:
            return pattern % (h, m, s)
        
    return ('%d days, ' + pattern) % (d, h, m, s)

##############################################################################
#                              EDF READER                                    #
##############################################################################
class EDFReader():
    """
    This class will read the edf file format.
    
    :param fname: filename to read.
    :type fname: string
        
    """
    def __init__(self, fname=None):
        self.fname = None
        self.meas_info = None
        self.chan_info = None
        self.calibrate = None
        self.offset    = None
        if fname:
            self.fname = fname
            self.readHeader()

    def open(self, fname):

        self.fname = fname
        
        return self.readHeader()
    
    def readHeader(self):
        try:
            # the following is copied over from MNE-Python and subsequently modified
            # to more closely reflect the native EDF standard
            meas_info = {}
            chan_info = {}
            filen = os.path.basename(self.fname).replace(' ','')
            
            if (self.fname.lower().endswith(".edf")):
                fid=open(self.fname, "r+b")
            elif (self.fname.lower().endswith(".edfz")) or (self.fname.lower().endswith(".edf.gz")):
                fid=gzip.open(self.fname, "rb")
            
            assert(fid.tell() == 0)
            meas_info['magic'] = fid.read(8).strip().decode()
            subject_id = fid.read(80).strip().decode()  # subject id
            meas_info['subject_code'] = subject_id.split(' ')[0]
            meas_info['gender'] = subject_id.split(' ')[1]
            meas_info['birthdate'] = subject_id.split(' ')[2]
            
            meas_info['subject_id'] = subject_id.split(' ')[-1]
            
            meas_info['firstname'] = None
            meas_info['lastname'] = None
            if not any(substring in meas_info['subject_id'].lower() for substring in {'x,x','x_x','x'}):
                meas_info['firstname'] = meas_info['subject_id'].replace('_',',').replace('-',',').split(',')[-1]
                meas_info['lastname'] = meas_info['subject_id'].replace('_',',').replace('-',',').split(',')[0]
                if meas_info['lastname'] == 'sub':
                    meas_info['lastname']=meas_info['firstname']
                    meas_info['firstname']='sub'
            else:
                if any(substring in filen for substring in {'~','_'}) and not filen.startswith('sub'):
                    firstname = filen.replace('_',' ').replace('~',' ').split()[1]
                    meas_info['firstname'] = firstname if firstname.lower() != 'x' else None
                    lastname = filen.replace('_',' ').replace('~',' ').split()[0]
                    meas_info['lastname'] = lastname if lastname.lower() != 'x' else None
            
            meas_info['recording_id'] = fid.read(80).strip().decode()  # recording id
            
            day, month, year = [int(x) for x in re.findall('(\d+)', fid.read(8).decode())]
            hour, minute, second = [int(x) for x in re.findall('(\d+)', fid.read(8).decode())]
            meas_info['day'] = day
            meas_info['month'] = month
            meas_info['year'] = year
            meas_info['hour'] = hour
            meas_info['minute'] = minute
            meas_info['second'] = second
            meas_info['meas_date'] = datetime.datetime(year + 2000, month, day, hour, minute, second).strftime('%Y-%m-%d %H:%M:%S')
            meas_info['data_offset'] = header_nbytes = int(fid.read(8).decode())
        
            subtype = fid.read(44).strip().decode()[:5]
            if len(subtype) > 0:
                meas_info['subtype'] = subtype
            else:
                meas_info['subtype'] = os.path.splitext(filen)[1][1:].lower()
        
            if meas_info['subtype'] in ('24BIT', 'bdf'):
                meas_info['data_size'] = 3  # 24-bit (3 byte) integers
            else:
                meas_info['data_size'] = 2  # 16-bit (2 byte) integers
        
            meas_info['n_records'] = int(fid.read(8).decode())
        
            # record length in seconds
            record_length = float(fid.read(8).decode())
            if record_length == 0:
                meas_info['record_length'] = record_length = 1.
                warnings.warn('Headermeas_information is incorrect for record length. '
                              'Default record length set to 1.')
            else:
                meas_info['record_length'] = record_length
            meas_info['nchan'] = nchan = int(fid.read(4).decode())
            
            channels = list(range(nchan))
            chan_info['ch_names'] = [fid.read(16).strip().decode() for ch in channels]
            chan_info['transducers'] = [fid.read(80).strip().decode() for ch in channels]
            chan_info['units'] = [fid.read(8).strip().decode() for ch in channels]
            chan_info['physical_min'] = np.array([float(fid.read(8).decode()) for ch in channels])
            chan_info['physical_max'] = np.array([float(fid.read(8).decode()) for ch in channels])
            chan_info['digital_min'] = np.array([float(fid.read(8).decode()) for ch in channels])
            chan_info['digital_max'] = np.array([float(fid.read(8).decode()) for ch in channels])
            prefiltering = [fid.read(80).strip().decode() for ch in channels][:-1]
            highpass = np.ravel([re.findall('HP:\s+(\w+)', filt) for filt in prefiltering])
            lowpass = np.ravel([re.findall('LP:\s+(\w+)', filt) for filt in prefiltering])
            high_pass_default = 0.
            if highpass.size == 0:
                meas_info['highpass'] = high_pass_default
            elif all(highpass):
                if highpass[0] == 'NaN':
                    meas_info['highpass'] = high_pass_default
                elif highpass[0] == 'DC':
                    meas_info['highpass'] = 0.
                else:
                    meas_info['highpass'] = float(highpass[0])
            else:
                meas_info['highpass'] = float(np.max(highpass))
                warnings.warn('Channels contain different highpass filters. '
                              'Highest filter setting will be stored.')
            
            if lowpass.size == 0:
                meas_info['lowpass'] = None
            elif all(lowpass):
                if lowpass[0] == 'NaN':
                    meas_info['lowpass'] = None
                else:
                    meas_info['lowpass'] = float(lowpass[0])
            else:
                meas_info['lowpass'] = float(np.min(lowpass))
                warnings.warn('%s' % ('Channels contain different lowpass filters.'
                                      ' Lowest filter setting will be stored.'))
            # number of samples per record
            chan_info['n_samps'] = n_samps = np.array([int(fid.read(8).decode()) for ch in channels])
            meas_info['sampling_frequency'] = int(chan_info['n_samps'][0]/meas_info['record_length'])
            
            fid.read(32 *meas_info['nchan']).decode()  # reserved
            assert fid.tell() == header_nbytes
            if meas_info['n_records']==-1:
                # this happens if the n_records is not updated at the end of recording
                tot_samps = (os.path.getsize(self.fname)-meas_info['data_offset'])/meas_info['data_size']
                meas_info['n_records'] = tot_samps/sum(n_samps)
            
            fid.close()
                
            self.calibrate = (chan_info['physical_max'] - chan_info['physical_min'])/(chan_info['digital_max'] - chan_info['digital_min'])
            self.offset =  chan_info['physical_min'] - self.calibrate * chan_info['digital_min']
            
            chan_info['calibrate'] = self.calibrate
            chan_info['offset'] = self.offset
            
            for ch in channels:
                if self.calibrate[ch]<0:
                    self.calibrate[ch] = 1
                    self.offset[ch] = 0
            
            self.header = {}
            self.header['meas_info'] = meas_info
            self.header['chan_info'] = chan_info
            
            self.meas_info = meas_info
            self.chan_info = chan_info
            
            tal_indx = [i for i,x in enumerate(self.header['chan_info']['ch_names']) if x.endswith('Annotations')][0]
            
            self.header['meas_info']['millisecond'] = meas_info['millisecond'] = self.read_annotation_block(0, tal_indx)[0][0][0]
            return self.header
        except Exception as e:
            self.header = {}
            ex_type,ex_value,ex_traceback = sys.exc_info()
            
            # Extract unformatter stack traces as tuples
            trace_back = traceback.extract_tb(ex_traceback)
            
            # Format stacktrace
            stack_trace = ["Error happened"]
            str_msg = ""
            for trace in trace_back:
                stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
                str_msg += f"\r\nFile : {trace[0]} , Line : {trace[1]}, Func.Name : {trace[2]}, Message : {trace[3]}"
            # print("Exception type : %s " % ex_type.__name__)
            # print("Exception message : %s" %ex_value)
            # print("Stack trace : %s" %stack_trace)

            stack_trace.append("")
            stack_trace.append("The error is due to checking input file:")
            stack_trace.append(f"\t\t{self.fname}")
            warningBox(stack_trace)
            return 
    
    def _read_annotations_apply_offset(self, triggers):
        events = []
        offset = 0.
        for k, ev in enumerate(triggers):
            onset = float(ev[0]) + offset
            duration = float(ev[2]) if ev[2] else 0
            for description in ev[3].split('\x14')[1:]:
                if description:
                    events.append([onset, duration, description, ev[4]])
                elif k==0:
                    # "The startdate/time of a file is specified in the EDF+ header
                    # fields 'startdate of recording' and 'starttime of recording'.
                    # These fields must indicate the absolute second in which the
                    # start of the first data record falls. So, the first TAL in
                    # the first data record always starts with +0.X2020, indicating
                    # that the first data record starts a fraction, X, of a second
                    # after the startdate/time that is specified in the EDF+
                    # header. If X=0, then the .X may be omitted."
                    offset = -onset
                    
        return events if events else list()
    
    def read_annotation_block(self, block, tal_indx):
        pat = '([+-]\\d+\\.?\\d*)(\x15(\\d+\\.?\\d*))?(\x14.*?)\x14\x00'
        assert(block>=0)
        data = []
        
        if (self.fname.lower().endswith(".edf")):
            with open(self.fname, 'rb') as fid:
                assert(fid.tell() == 0)
                blocksize = np.sum(self.header['chan_info']['n_samps']) * self.header['meas_info']['data_size']
                fid.seek(np.int64(self.header['meas_info']['data_offset']) + np.int64(block) * np.int64(blocksize))
                read_idx = 0
                for i in range(self.header['meas_info']['nchan']):
                    read_idx += np.int64(self.header['chan_info']['n_samps'][i]*self.header['meas_info']['data_size'])
                    buf = fid.read(np.int64(self.header['chan_info']['n_samps'][i]*self.header['meas_info']['data_size']))
                    if i==tal_indx:
                        raw = re.findall(pat, buf.decode('latin-1'))
                        if raw:
                            data.append(list(map(list, [x+(block,) for x in raw])))
        elif (self.fname.lower().endswith(".edfz")) or (self.fname.lower().endswith(".edf.gz")):
            with gzip.open(self.fname, 'rb') as fid:
                assert(fid.tell() == 0)
                blocksize = np.sum(self.header['chan_info']['n_samps']) * self.header['meas_info']['data_size']
                fid.seek(np.int64(self.header['meas_info']['data_offset']) + np.int64(block) * np.int64(blocksize))
                read_idx = 0
                for i in range(self.header['meas_info']['nchan']):
                    read_idx += np.int64(self.header['chan_info']['n_samps'][i]*self.header['meas_info']['data_size'])
                    buf = fid.read(np.int64(self.header['chan_info']['n_samps'][i]*self.header['meas_info']['data_size']))
                    if i==tal_indx:
                        raw = re.findall(pat, buf.decode('latin-1'))
                        if raw:
                            data.append(list(map(list, [x+(block,) for x in raw])))

        return data
    
    def overwrite_annotations(self, events, identity_idx, tal_indx, strings, action):
        pat = '([+-]\\d+\\.?\\d*)(\x15(\\d+\\.?\\d*))?(\x14.*?)\x14\x00'
        indexes = []
        for ident in identity_idx:
            block_chk = events[ident][-1]
            replace_idx = [i for i,x in enumerate(strings) if x.lower() in events[ident][2].lower()]
            for irep in replace_idx:
                assert(block_chk>=0)
                new_block=[]
                
                if (self.fname.lower().endswith(".edf")):
                    fid=open(self.fname, "rb")
                elif (self.fname.lower().endswith(".edfz")) or (self.fname.lower().endswith(".edf.gz")):
                    fid=gzip.open(self.fname, "rb")
                
                assert(fid.tell() == 0)
                blocksize = np.sum(self.header['chan_info']['n_samps']) * self.header['meas_info']['data_size']
                fid.seek(np.int64(self.header['meas_info']['data_offset']) + np.int64(block_chk) * np.int64(blocksize))
                for i in range(self.header['meas_info']['nchan']):
                    buf = fid.read(np.int64(self.header['chan_info']['n_samps'][i]*self.header['meas_info']['data_size']))
                    if i==tal_indx:
                        if action == 'replace':
                            new_block = buf.lower().replace(bytes(strings[irep],'latin-1').lower(), bytes(''.join(np.repeat('X', len(strings[irep]))),'latin-1'))
                            events[ident][2] = events[ident][2].lower().replace(strings[irep].lower(), ''.join(np.repeat('X', len(strings[irep]))))
                            assert(len(new_block)==len(buf))
                        
                        elif action == 'replaceWhole':
                            _idx = [i for i,x in enumerate(strings.keys()) if x.lower() in events[ident][2].lower()]
                            replace_string = list(strings.values())[_idx[0]]
                            new_block = buf.lower().replace(bytes(events[ident][2],'latin-1').lower(), bytes(replace_string,'latin-1'))
                            events[ident][2] = replace_string
                            new_block = new_block+bytes('\x00'*(len(buf)-len(new_block)),'latin-1')
                            assert(len(new_block)==len(buf))
                        
                        elif action == 'remove':
                            raw = re.findall(pat, buf.decode('latin-1'))[0][0] +'\x14\x14'
                            new_block = raw + ('\x00'*(len(buf)-len(raw)))
                            indexes.append(ident)
                            assert(len(new_block)==len(buf))
                fid.close()
                
                if new_block:
                    
                    if (self.fname.lower().endswith(".edf")):
                        fid=open(self.fname, "wb")
                    elif (self.fname.lower().endswith(".edfz")) or (self.fname.lower().endswith(".edf.gz")):
                        fid=gzip.open(self.fname, "wb")

                    read_idx = 0
                    for i in range(self.header['meas_info']['nchan']):
                        assert(fid.tell() == 0)
                        blocksize = np.sum(self.header['chan_info']['n_samps']) * self.header['meas_info']['data_size']
                        fid.seek(np.int64(self.header['meas_info']['data_offset']) + np.int64(block_chk) * np.int64(blocksize))

                        read_idx += np.int64(self.header['chan_info']['n_samps'][i]*self.header['meas_info']['data_size'])
                        buf = fid.read(np.int64(self.header['chan_info']['n_samps'][i]*self.header['meas_info']['data_size']))
                        if i==tal_indx:
                            back = fid.seek(-np.int64(self.header['chan_info']['n_samps'][i]*self.header['meas_info']['data_size']), 1)
                            assert(fid.tell()==back)
                            fid.write(new_block)
                    fid.close()
        
        if indexes:
            for index in sorted(indexes, reverse=True):
                del events[index]
        
        return events
    
    def annotations(self):
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
        self.header = file_in.open(self.fname)
        
        tal_indx = [i for i,x in enumerate(self.header['chan_info']['ch_names']) if x.endswith('Annotations')][0]
        
        start_time = 0
        end_time = self.header['meas_info']['n_records']*self.header['meas_info']['record_length']
        
        begsample = int(self.header['meas_info']['sampling_frequency']*float(start_time))
        endsample = int(self.header['meas_info']['sampling_frequency']*float(end_time))
        
        n_samps = max(set(list(self.header['chan_info']['n_samps'])), key = list(self.header['chan_info']['n_samps']).count)
        
        begblock = int(np.floor((begsample) / n_samps))
        endblock = int(np.floor((endsample) / n_samps))
        
        update_cnt = int((endblock+1)/10)
        annotations = []
        for block in range(begblock, endblock):
            data_temp = self.read_annotation_block(block, tal_indx)
            if data_temp:
                annotations.append(data_temp[0])
            if block == update_cnt and block < (endblock-(int((endblock+1)/20))):
                print('{}%'.format(int(np.ceil((update_cnt/endblock)*100))))
                update_cnt += int((endblock+1)/10)
        
        events = self._read_annotations_apply_offset([item for sublist in annotations for item in sublist])
        
        annotation_data = pd.DataFrame({})
        if events:
            fulldate = datetime.datetime.strptime(self.header['meas_info']['meas_date'], '%Y-%m-%d %H:%M:%S')
            for i, annot in enumerate(events):
                data_temp = {'onset': annot[0],
                             'duration': annot[1],
                             'time_abs': (fulldate + datetime.timedelta(seconds=annot[0]+float(self.header['meas_info']['millisecond']))).strftime('%H:%M:%S.%f'),
                             'time_rel': sec2time(annot[0], 6),
                             'event': annot[2]}
                annotation_data = pd.concat([annotation_data, pd.DataFrame([data_temp])], axis = 0)
        
        return annotation_data
    
    def get_montage(self, fname):
    
        ignore_keys={'IChannelId','IInputId','ISiteId','ITypeId','OChannelId','OTypeId','GroupId'}
        
        mtg_file = np.fromfile(fname, dtype='uint8')
        mtg_file_tmp = "".join([struct.unpack('s', x)[0].decode('ISO-8859-1') for x in mtg_file])
        mtg_file_tmp = re.findall(r'\(.\(..*?\)\)', mtg_file_tmp)
        
        #chan info
        chans_info = [x.replace('(.','(') for x in mtg_file_tmp if x.startswith('(.(."ChanIndex"')]
        
        chan_info=[]
        for ichan in range(len(chans_info)):
            chan_info_tmp=[eval(re.findall(r'\(.*?\)',chans_info[ichan])[0].replace('((','('))]+[eval(x) for x in re.findall(r'\(.*?\)',chans_info[ichan])[2:] if not any( y in x for y in ignore_keys)]
            chan_info.append({key: value for (key, value) in chan_info_tmp})
            
        chan_info=pd.DataFrame(chan_info)
        chan_info=chan_info[chan_info.To_Name != 0].reset_index(drop=True)
        
        chan_info['ChanIndex'] = chan_info.index
        
        return chan_info.loc[:,['ChanIndex','To_Name']].values
    
    def chnames_update(self, channel_file, bids_settings, write=False):
        
        if (self.fname.lower().endswith(".edf")):
            fid=open(self.fname, "rb")
        elif (self.fname.lower().endswith(".edfz")) or (self.fname.lower().endswith(".edf.gz")):
            fid=gzip.open(self.fname, "rb")
        
        with open(self.fname, 'rb') as fid:
            fid.seek(252)
            nchan = int(fid.read(4).strip().decode())
            channels = list(range(nchan))
            ch_names_orig= [fid.read(16).strip().decode() for ch in channels]
        
        
        chan_idx = [i for i, x in enumerate(ch_names_orig) if not any(x.startswith(substring) for substring in list(bids_settings['natus_info']['ChannelInfo'].keys()))]
        
        if channel_file.endswith('.mtg'):
            chan_label_new = self.get_montage(channel_file)
        else:
            chan_label_new = np.genfromtxt(channel_file, dtype='str')
        
        if len(chan_label_new) >1:
            chan_label_new=[x[1] if isinstance(x,np.ndarray) else x.split(',')[1] for x in chan_label_new]
        
        if len(chan_label_new)<=len(chan_idx):
            replace_chan = [str(x) for x in list(range(len(chan_label_new)+1,len(chan_idx)+1))]
            chan_label_new.extend([''.join(list(item)) for item in list(zip(['C']*len(replace_chan), replace_chan))])
            assert len(chan_label_new)==len(chan_idx)
        
        elif len(chan_label_new)>len(chan_idx):
            add_chans = (len(chan_label_new)-len(chan_idx))+1
            if not chan_idx:
                chan_idx=list(range(1, add_chans))
            else:
                chan_idx+=list(range(chan_idx[-1]+1, (chan_idx[-1]+add_chans)))
            assert len(chan_label_new)==len(chan_idx)
        
        ch_names_new=ch_names_orig
        for (index, replacement) in zip(chan_idx, chan_label_new):
            ch_names_new[index] = replacement
        
        if write:
            if (self.fname.lower().endswith(".edf")):
                with open(self.fname, 'r+b') as fid:
                    assert(fid.tell() == 0)
                    fid.seek(256)
                    for ch in ch_names_new:
                        fid.write(padtrim(ch,16))
            elif (self.fname.lower().endswith(".edfz")) or (self.fname.lower().endswith(".edf.gz")):
                with gzip.open(self.fname, 'r+b') as fid:
                    assert(fid.tell() == 0)
                    fid.seek(256)
                    for ch in ch_names_new:
                        fid.write(padtrim(ch,16))
        
        return ch_names_new
        
    def readBlock(self, block):
        assert(block>=0)
        data = []
        
        for i in range(self.header['meas_info']['nchan']):
            if (self.fname.lower().endswith(".edf")):
                fid=open(self.fname, "rb")
            elif (self.fname.lower().endswith(".edfz")) or (self.fname.lower().endswith(".edf.gz")):
                fid=gzip.open(self.fname, "rb")
            assert(fid.tell() == 0)
            blocksize = np.sum(self.header['chan_info']['n_samps']) * self.header['meas_info']['data_size']
            fid.seek(self.header['meas_info']['data_offset'] + block * blocksize)
            buf = fid.read(self.header['chan_info']['n_samps'][i]*self.header['meas_info']['data_size'])
            raw = np.asarray(unpack('<{}h'.format(self.header['chan_info']['n_samps'][i]), buf), dtype=np.float32)
            raw *= self.calibrate[i]
            raw += self.offset[i]  # FIXME I am not sure about the order of calibrate and offset
            data.append(raw)
            fid.close()
        
        return (data)

##############################################################################
#                             BIDS HELPER                                    #
##############################################################################
class bidsHelper():
    def __init__(self, subject_id=None, session_id=None, task_id=None, run_num=None, 
              kind=None, suffix=None, output_path=None, bids_settings=None, make_sub_dir=False):
        
        self.subject_id = subject_id
        self.session_id = session_id
        self.task_id = task_id
        self.run_num = run_num
        self.kind = kind
        self.suffix = suffix
        self.output_path = output_path
        self.bids_settings = bids_settings
        
        if make_sub_dir:
            self.make_bids_folders(make_dir=True)
            
    def write_scans(self, file_name, file_info_run, date_offset):
        # Add scans json file for each subject
        self.scans_fname = self.make_bids_filename(suffix='scans.tsv', exclude_ses=True, exclude_task=True, exclude_run=True, 
                                             path_override=os.path.join(self.output_path,self.subject_id))
        self.scans_json_fname = self.make_bids_filename(suffix='scans.json', exclude_ses=True, exclude_task=True, exclude_run=True, 
                                             path_override=os.path.join(self.output_path,self.subject_id))
        if not os.path.exists(self.scans_json_fname):
            self._scans_json()
        
        self._scans_data(file_name, file_info_run, date_offset)
        
    def write_channels(self, file_info_run, overwrite=False, verbose=False):
        
        self.channels_fname = self.make_bids_filename(suffix='channels.tsv')
        self._channels_data(file_info_run, overwrite, verbose)
    
    def write_sidecar(self, file_info_run, overwrite=False, verbose=False):
        
        self.sidecar_fname = self.make_bids_filename(suffix=self.kind + '.json')
        self._sidecar_json(file_info_run, overwrite, verbose)
    
    def write_electrodes(self, file_info_run, coordinates, overwrite=False, verbose=False):
        
        self.electrodes_fname = self.make_bids_filename(suffix='electrodes.tsv',
                                                    exclude_task=True, exclude_run=True)
        self._electrodes_data(file_info_run, coordinates, overwrite, verbose)
    
    def write_participants(self, data=None, return_fname=False):
        
        self.participants_fname = self.make_bids_filename(suffix='participants.tsv', exclude_sub=True, exclude_ses=True, 
                                                    exclude_task=True, exclude_run=True, path_override=self.output_path)
        self.participants_json_fname = self.make_bids_filename(suffix='participants.json', exclude_sub=True, exclude_ses=True, 
                                                    exclude_task=True, exclude_run=True, path_override=self.output_path)
        if not return_fname:
            self._participants_data(data)
            if not os.path.exists(self.participants_json_fname):
                self._participants_json()
        else:
            return self.participants_fname
    
    def write_dataset(self, return_fname=False):
        
        self.dataset_fname = self.make_bids_filename(suffix='dataset_description.json', exclude_sub=True, exclude_ses=True, 
                                                    exclude_task=True, exclude_run=True, path_override=self.output_path)
        if not return_fname:
            self._dataset_json()
        else:
            return self.dataset_fname
        
    def make_bids_filename(self, suffix, exclude_sub=False, exclude_ses=False, exclude_task=False, exclude_run=False, path_override=None):
        """
        Constructs a BIDsified filename.
        
        :param subject_id: Subject identifier.
        :type subject_id: string or None
        :param session_id: Session identifier.
        :type session_id: string or None
        :param run_num: Run identifier.
        :type run_num: string or None
        :param suffix: Extension for the file.
        :type suffix: string
        :param prefix: Path to where the file is to be saved.
        :type prefix: string
        
        :return filename: BIDsified filename.
        :type filename: string
        
        """
        ses=self.session_id
        if exclude_ses:
            ses=None
        elif isinstance(self.session_id, str):
            if 'ses' in self.session_id:
                ses=self.session_id.split('-')[1]
            
        if exclude_task:
            task=None
        else:
            task=self.task_id
        
        if exclude_run:
            run=None
        else:
            run=self.run_num
            
        order = OrderedDict([('ses', ses if ses is not None else None),
                             ('task', task if task is not None else None),
                             ('run', run if run is not None else None)])
    
        filename = []
        if not exclude_sub:
            if self.subject_id is not None:
                filename.append(self.subject_id)
                
        for key, val in order.items():
            if val is not None:
                filename.append('%s-%s' % (key, val))
    
        if isinstance(suffix, str):
            filename.append(suffix)
    
        filename = '_'.join(filename)
        if path_override is not None:
            filename = os.path.join(path_override, filename)
        else:
            prefix=self.make_bids_folders()
            filename = os.path.join(prefix, filename)
            
        return filename

    def make_bids_folders(self, path_override=None, make_dir=False, overwrite=False):
        """
        Constructs a BIDsified folder structure.
        
        :param subject_id: Subject identifier.
        :type subject_id: string or None
        :param session_id: Session identifier.
        :type session_id: string or None
        :param kind: Type of input data (e.g. anat, ieeg etc.).
        :type kind: string
        :param output_path: Path to where the file is to be saved.
        :type output_path: string
        :param make_dir: Make the directory
        :type make_dir: boolean
        :param overwrite: If duplicate data is present in the output directory overwrite it.
        :type overwrite: boolean
        
        :return path: BIDsified folder path.
        :type path: string
        
        """
        path = []
        path.append(self.subject_id)
            
        if isinstance(self.session_id, str):
            if 'ses' not in self.session_id:
                path.append('ses-%s' % self.session_id)
            else:
                path.append(self.session_id)
        
        if path_override is None:
            if isinstance(self.kind, str):
                path.append(self.kind)
        
        path = os.path.join(*path)
        if path_override is not None:
            path = os.path.join(path_override, path)
        else:
            path = os.path.join(self.output_path, path)
            
        if make_dir == True:
            if not os.path.exists(path):
                os.makedirs(path)
            elif overwrite:
                os.makedirs(path)
                
        return path
    
    def _write_tsv(self, fname, data, float_form=None, overwrite=False, verbose=False, append = False):
        """
        Writes input dataframe to a .tsv file
        
        :param fname: Filename given to the output tsv file.
        :type fname: string
        :param data: Dataframe containing the information to write.
        :type data: dataframe
        :param overwrite: If duplicate data is present in the output directory overwrite it.
        :type overwrite: boolean
        :param verbose: Print out process steps.
        :type verbose: boolean
        :param append: Append data to file if it exists.
        :type append: boolean
    
        """
        if sys.platform == 'win32':
            if os.path.exists(fname) and not overwrite:
                pass
            if os.path.exists(fname) and append:
                with open(fname,'a') as f:
                    if float_form is not None:
                        data.to_csv(f, sep='\t', index=False, header = False, na_rep='n/a', mode='a', float_format=float_form)
                    else:
                        data.to_csv(f, sep='\t', index=False, header = False, na_rep='n/a', mode='a')
                with open(fname) as f:
                    lines = f.readlines()
                    last = len(lines) - 1
                    lines[last] = lines[last].replace('\r','').replace('\n','')
                with open(fname, 'w') as wr:
                    wr.writelines(lines)
            else:
                data1 = data.iloc[0:len(data)-1]
                data2 = data.iloc[[len(data)-1]]
                data1.to_csv(fname, sep='\t', encoding='utf-8', index = False)
                if float_form is not None:
                    data2.to_csv(fname, sep='\t', encoding='utf-8', index=False, header = False, na_rep='n/a', mode='a', float_format=float_form)
                else:
                    data2.to_csv(fname, sep='\t', encoding='utf-8', index=False, header = False, na_rep='n/a', mode='a')
        else:
            if os.path.exists(fname) and not overwrite:
                pass
            if os.path.exists(fname) and append:
                with open(fname,'a') as f:
                    if float_form is not None:
                        data.to_csv(f, sep='\t', encoding='utf-8', index=False, header = False, na_rep='n/a', mode='a', float_format=float_form)
                    else:
                        data.to_csv(f, sep='\t', encoding='utf-8', index=False, header = False, na_rep='n/a', mode='a')
            else:
                data1 = data.iloc[0:len(data)-1]
                data2 = data.iloc[[len(data)-1]]
                data1.to_csv(fname, sep='\t', encoding='utf-8', index = False)
                if float_form is not None:
                    data2.to_csv(fname, sep='\t', encoding='utf-8', header= False, index = False, na_rep='n/a', mode='a',float_format=float_form)
                else:
                    data2.to_csv(fname, sep='\t', encoding='utf-8', header= False, index = False, na_rep='n/a', mode='a')

    def _write_json(self, data, fname, overwrite=False, verbose=False):
        """
        Writes input data to a .json file
        
        :param fname: Filename given to the output json file.
        :type fname: string
        :param dictionary: Dictionary containing the information to write.
        :type dictionary: dictionary
        :param overwrite: If duplicate data is present in the output directory overwrite it.
        :type overwrite: boolean
        :param verbose: Print out process steps.
        :type verbose: boolean
    
        """
        json_output = json.dumps(data, indent=4)
        with open(fname, 'w') as fid:
            fid.write(json_output)
            fid.write('\n')
    
        if verbose is True:
            print(os.linesep + "Writing '%s'..." % fname + os.linesep)
            print(json_output)
    
    def _dataset_json(self):
        """
        Constructs a dataset description JSON file.
        
        :param dataset_fname: Filename for the BIDS dataset description.
        :type dataset_fname: string
        
        """
        info_dataset_json = OrderedDict([
            ('Name', self.bids_settings['json_metadata']['DatasetName']),
            ('BIDSVersion', ''),
            ('License', ''),
            ('Authors', self.bids_settings['json_metadata']['Experimenter'][0]),
            ('Acknowledgements', 'say here what are your acknowledgments'),
            ('HowToAcknowledge', 'say here how you would like to be acknowledged'),
            ('Funding', ["list your funding sources"]),
            ('ReferencesAndLinks', ["a data paper", "a resource to be cited when using the data"]),
            ('DatasetDOI', '')])
        
        self._write_json(info_dataset_json, self.dataset_fname)

    def _participants_json(self):
        """
        Constructs a participant description JSON file.
        
        :param participants_fname: Filename for the BIDS participant description.
        :type participants_fname: string
        
        """
        info_participant_json = OrderedDict([
            ('age', 
                     {'Description': 'age of the participants.', 
                      'Units': 'years.'}),
            ('sex', 
                     {'Description': 'sex of the participants.', 
                      'Levels': {'m': 'male',
                                  'f': 'female'}}),
            ('group', 
                     {'Description': 'group the patient belongs to', 
                      })
            ])
    
        self._write_json(info_participant_json, self.participants_json_fname)
    
    def _participants_data(self, file_info_sub):
        """
        Constructs a participant tsv file.
        
        :param subject_id: Subject identifier.
        :type subject_id: string or None
        :param file_info_sub: File header information from subjects recordings.
        :type file_info_sub: dictionary
        :param participants_fname: Filename for the BIDS participant description.
        :type participants_fname: string
        
        """
        if self.subject_id is None:
            with open(self.participants_fname, 'w') as writeFile:
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
                              ('participant_id', self.subject_id if self.subject_id is not None else ''),
                              ('age', age if age else 'n/a'),
                              ('sex', gender if gender else 'n/a'),
                              ('group', 'patient')]), index= [0])
            
            self._write_tsv(self.participants_fname, df, overwrite=False, verbose=False, append = True)

    def _scans_json(self):
        """
        Constructs a scans JSON file containing list of all data files stored in patient directory.
        
        :param scans_fname: Filename for the scans JSON file.
        :type scans_fname: string
        
        """
        info_scans_json = OrderedDict([
            ('duration', 
                     {'Description': 'total duration of the recording.', 
                      'Units': 'hours.'}),
            ('edf_type', 
                     {'Description': 'type of EDF file.', 
                      'Units': 'EDF+D or EDF+C.'})
            ])
    
        self._write_json(info_scans_json, self.scans_json_fname)
    
    def _scans_data(self, file_name, file_info_run, date_offset):
        """
        Constructs a scans tsv file containing list of all data files stored in patient directory.
        
        :param file_name: Filename for the specific recording.
        :type file_name: string
        :param file_info_run: File header information for specific recording.
        :type file_info_run: dictionary
        :param scans_fname: Filename for the scans tsv file.
        :type scans_fname: string
        
        """
        date = datetime.datetime.strptime(file_info_run['Date'], '%Y-%m-%d')
        if date_offset:
            date = date - datetime.timedelta(5856)
        acq_time = 'T'.join([date.strftime('%Y-%m-%d'), file_info_run['Time']])
        
        df = pd.DataFrame(OrderedDict([
                      ('filename', file_name),
                      ('acq_time', acq_time),
                      ('duration', file_info_run['TotalRecordTime']),
                      ('edf_type', file_info_run['EDF_type'])
                      ]), index=[0])
        
        self._write_tsv(self.scans_fname, df, float_form='%.3f', overwrite=False, verbose=False, append = True)
    
    def _electrodes_data(self, file_info_run, coordinates, overwrite, verbose):
        """
        Constructs a tsv file containing electrode information for recording.
        
        :param file_info_run: File header information for specific recording.
        :type file_info_run: dictionary
        :param electrodes_fname: Filename for the electrode tsv file.
        :type electrodes_fname: string
        :param coordinates: List of electrode coordinates (x,y,z).
        :type coordinates: list or None
        :param electrode_imp: Lst of electrode impedances.
        :type electrode_imp: list or None
        :param overwrite: If duplicate data is present in the output directory overwrite it.
        :type overwrite: boolean
        :param verbose: Print out process steps.
        :type verbose: boolean
        
        """
        include_chans = ['SEEG','EEG']
        chan_idx = [i for i, x in enumerate(list(file_info_run['ChanInfo'].keys())) if x in include_chans]
        check_cnt=0
        for ichan in chan_idx:
            check_cnt += file_info_run['ChanInfo'][list(file_info_run['ChanInfo'].keys())[ichan]]['ChannelCount']
            
        if check_cnt==0:
            chan_idx = [i for i, x in enumerate(list(file_info_run['ChanInfo'].keys())) if x in {'C'}]
            
            
        mainDF = pd.DataFrame([])
        for ichan in range(len(chan_idx)):
            info_temp = file_info_run['ChanInfo'][list(file_info_run['ChanInfo'].keys())[chan_idx[ichan]]]
            
            if 'SEEG' in info_temp['Type']:
                values = []
                material = []
                manu = []
                size = []
                typ = []
                for item in info_temp['ChanName']:
                    value_temp = 'n/a'
                    if [x for x in file_info_run['Groups'] if item.startswith(x)]:
                        value_temp = [x for x in file_info_run['Groups'] if item.startswith(x)][0]
                    values.append(value_temp)
                    material.append(self.bids_settings['natus_info']['iEEGElectrodeInfo']['Material'])
                    manu.append(self.bids_settings['natus_info']['iEEGElectrodeInfo']['Manufacturer'])
                    size.append(self.bids_settings['natus_info']['iEEGElectrodeInfo']['Diameter'])
                    typ.append('depth')
            elif 'EEG' in info_temp['Type']:
                values = []
                material = []
                manu = []
                size = []
                typ = []
                for item in info_temp['ChanName']:
                    value_temp = 'n/a'
                    if [x for x in file_info_run['Groups'] if item.startswith(x)]:
                        value_temp = [x for x in file_info_run['Groups'] if item.startswith(x)][0]
                    
                    values.append(value_temp)
                    material.append(self.bids_settings['natus_info']['EEGElectrodeInfo']['Material'])
                    manu.append(self.bids_settings['natus_info']['EEGElectrodeInfo']['Manufacturer'])
                    size.append(self.bids_settings['natus_info']['EEGElectrodeInfo']['Diameter'])
                    typ.append('scalp')
                    
            else:
                values = np.repeat('n/a', len(info_temp['ChanName']))
                material = np.repeat('n/a', len(info_temp['ChanName']))
                manu = np.repeat('n/a', len(info_temp['ChanName']))
                size = np.repeat('n/a', len(info_temp['ChanName']))
                typ = np.repeat('n/a', len(info_temp['ChanName']))
    
            df = pd.DataFrame(OrderedDict([
                          ('name', info_temp['ChanName']),
                          ('x', coordinates[0] if coordinates is not None else 'n/a'),
                          ('y', coordinates[1] if coordinates is not None else 'n/a'),
                          ('z', coordinates[2] if coordinates is not None else 'n/a'),
                          ('size', size),
                          ('type', typ),
                          ('material', material),
                          ('manufacturer', manu)]))
            
            mainDF = pd.concat([mainDF, df], ignore_index = True, axis = 0) 
            
        df_electrodes = []
        for key, val in mainDF.items():
            if val is not None:
                df_electrodes.append((key, val))
                
        df_electrodes = pd.DataFrame(OrderedDict(df_electrodes))
        
        self._write_tsv(self.electrodes_fname, df_electrodes, overwrite=False, verbose=False, append = True)

    def _channels_data(self, file_info_run, overwrite, verbose):
        """
        Constructs a tsv file containing channel information for recording.
        
        :param file_info_run: File header information for specific recording.
        :type file_info_run: dictionary
        :param channels_fname: Filename for the channels tsv file.
        :type channels_fname: string
        :param overwrite: If duplicate data is present in the output directory overwrite it.
        :type overwrite: boolean
        :param verbose: Print out process steps.
        :type verbose: boolean
        
        """
        include_chans = ['SEEG','EEG']
        chan_idx = [i for i, x in enumerate(list(file_info_run['ChanInfo'].keys())) if x in include_chans]
        check_cnt=0
        for ichan in chan_idx:
            check_cnt += file_info_run['ChanInfo'][list(file_info_run['ChanInfo'].keys())[ichan]]['ChannelCount']
            
        if check_cnt==0:
            chan_idx = [i for i, x in enumerate(list(file_info_run['ChanInfo'].keys())) if x in {'C'}]
            
        mainDF = pd.DataFrame([])
        for ichan in chan_idx:
            info_temp = file_info_run['ChanInfo'][list(file_info_run['ChanInfo'].keys())[ichan]]
            
            if 'SEEG' in info_temp['Type']:
                values = []
                for item in info_temp['ChanName']:
                    value_temp = 'n/a'
                    if [x for x in file_info_run['Groups'] if item.startswith(x)]:
                        value_temp = [x for x in file_info_run['Groups'] if item.startswith(x)][0]
                    values.append(value_temp)
                    
            elif 'EEG' in info_temp['Type']:
                values = []
                for item in info_temp['ChanName']:
                    value_temp = 'n/a'
                    if [x for x in file_info_run['Groups'] if item.startswith(x)]:
                        value_temp = [x for x in file_info_run['Groups'] if item.startswith(x)][0]
                    
                    values.append(value_temp)
            else:
                values = np.repeat('n/a', len(info_temp['ChanName']))
    
            df = pd.DataFrame(OrderedDict([
                          ('name', info_temp['ChanName']),
                          ('type', np.repeat(info_temp['Type'], len(info_temp['ChanName']))),
                          ('units', info_temp['Unit']),
                          ('low_cutoff', np.repeat(file_info_run['Lowpass'] if file_info_run['Lowpass'] is not None else 'n/a', len(info_temp['ChanName']))),
                          ('high_cutoff', np.repeat(file_info_run['Highpass'] if file_info_run['Highpass'] is not None else 'n/a', len(info_temp['ChanName']))),
                          ('sampling_frequency', np.repeat(file_info_run['SamplingFrequency'], len(info_temp['ChanName']))),
                          ('notch', np.repeat('n/a',len(info_temp['ChanName']))),
                          ('reference', np.repeat('n/a',len(info_temp['ChanName']))),
                          ('group', values)]))
            
            mainDF = pd.concat([mainDF, df], ignore_index = True, axis = 0) 
        
        mainDF['units'] = mainDF['units'].str.replace('uV', 'μV')
        self._write_tsv(self.channels_fname, mainDF, overwrite=False, verbose=False, append = False)
    
    def _sidecar_json(self, file_info_run, overwrite, verbose):
        """
        Constructs a sidecar JSON file containing information about specific recording.
        
        :param file_info_run: File header information for specific recording.
        :type file_info_run: dictionary
        :param sidecar_fname: Filename for thesidecar JSON file.
        :type sidecar_fname: string
        :param overwrite: If duplicate data is present in the output directory overwrite it.
        :type overwrite: boolean
        :param verbose: Print out process steps.
        :type verbose: boolean
        
        """
        if 'iEEG' in file_info_run['RecordingType']:
            electrode_manu = self.bids_settings['natus_info']['iEEGElectrodeInfo']['Manufacturer']
        elif 'Scalp' in file_info_run['RecordingType']:
            electrode_manu = self.bids_settings['natus_info']['EEGElectrodeInfo']['Manufacturer']
        else:
            electrode_manu = 'n/a'
        
        task_name = file_info_run['RecordingLength']
                            
        if 'Ret' in file_info_run['Retro_Pro']:
            task_name = task_name + 'ret'
        
        info_sidecar_json = OrderedDict([
            ('TaskName', task_name),
            ('InstitutionName', self.bids_settings['json_metadata']['InstitutionName']),
            ('InstitutionAddress', self.bids_settings['json_metadata']['InstitutionAddress']),
            ('Manufacturer', self.bids_settings['natus_info']['Manufacturer']),
            ('ManufacturersModelName', self.bids_settings['natus_info']['ManufacturersModelName']),
            ('SamplingFrequency', file_info_run['SamplingFrequency']),
            ('HardwareFilters', 
                     {'HighpassFilter': {"Cutoff (Hz)": file_info_run['Highpass'] if file_info_run['Highpass'] is not None else 'n/a'}, 
                      'LowpassFilter': {"Cutoff (Hz)": file_info_run['Lowpass'] if file_info_run['Lowpass'] is not None else 'n/a'}}),
            ('SoftwareFilters', 'n/a'),
            ('EEGChannelCount', file_info_run['ChanInfo']['EEG']['ChannelCount'] if 'EEG' in list(file_info_run['ChanInfo'].keys()) else 0),
            ('EOGChannelCount', file_info_run['ChanInfo']['EOG']['ChannelCount'] if 'EOG' in list(file_info_run['ChanInfo'].keys()) else 0),
            ('ECGChannelCount', file_info_run['ChanInfo']['ECG']['ChannelCount'] if 'ECG' in list(file_info_run['ChanInfo'].keys()) else 0),
            ('EMGChannelCount', file_info_run['ChanInfo']['EMG']['ChannelCount'] if 'EMG' in list(file_info_run['ChanInfo'].keys()) else 0),
            ('ECOGChannelCount', file_info_run['ChanInfo']['ECOG']['ChannelCount'] if 'ECOG' in list(file_info_run['ChanInfo'].keys()) else 0),
            ('SEEGChannelCount', file_info_run['ChanInfo']['SEEG']['ChannelCount'] if 'SEEG' in list(file_info_run['ChanInfo'].keys()) else 0),
            ('MiscChannelCount', file_info_run['ChanInfo']['MISC']['ChannelCount'] if 'MISC' in list(file_info_run['ChanInfo'].keys()) else 0),
            ('TriggerChannelCount', file_info_run['ChanInfo']['TRIG']['ChannelCount'] if 'TRIG' in list(file_info_run['ChanInfo'].keys()) else 0),
            ('PowerLineFrequency', self.bids_settings['natus_info']['PowerLineFrequency']),
            ('RecordingDuration', file_info_run['TotalRecordTime']),
            ('RecordingType', 'continuous'),
            ('SubjectArtefactDescription', ''),
            ('iEEGPlacementScheme', ''),
            ('iEEGElectrodeGroups', ''),
            ('iEEGReference', ''),
            ('ElectrodeManufacturer', electrode_manu),
            ('ElectricalStimulationParameters', '')])
        
        if file_info_run['RecordingType'] == 'Scalp':
            info_sidecar_json['EEGPlacementScheme'] = info_sidecar_json.pop('iEEGPlacementScheme')
            info_sidecar_json['EEGReference'] = info_sidecar_json.pop('iEEGReference')
            del info_sidecar_json['SEEGChannelCount']
            del info_sidecar_json['iEEGElectrodeGroups']
            del info_sidecar_json['ECOGChannelCount']
            del info_sidecar_json['ElectrodeManufacturer']
            del info_sidecar_json['ElectricalStimulationParameters']
            
        self._write_json(info_sidecar_json, self.sidecar_fname, overwrite, verbose)

def get_file_info(raw_file_path_sub, bids_settings):
    """
    Extract header data from EDF file.
    
    :param raw_file_path_sub: Path to the raw data file for specific recording.
    :type raw_file_path_sub: string
    
    :return filesInfo: Dictionary containing header information for all files in specified path.
    :type filesInfo: dictionary
    
    """
    sub_dir = False

    file_list = [[x] for x in os.listdir(raw_file_path_sub) if x.lower().endswith('.edf')]
    if not file_list:
        sub_dir = True
        folders = [x for x in os.listdir(raw_file_path_sub) if os.path.isdir(os.path.join(raw_file_path_sub, x))]
        for ifold in folders:
            files = [os.path.sep.join([ifold, x]) for x in os.listdir(os.path.join(raw_file_path_sub, ifold)) if x.lower().endswith('.edf')]
            file_list.append(files)
            
    chan_label_filename = [x for x in os.listdir(raw_file_path_sub) if re.search('_labels', x, re.IGNORECASE)]
    
    filesInfo = []
    for ises in file_list:
        filesInfo_ses = []
        for ifile in ises:
            try:
                filen = os.path.join(raw_file_path_sub, ifile)
                file_in = EDFReader()
                header = file_in.open(filen)
                
                chan_label_file_ses = []
                if sub_dir:
                    chan_label_file_ses = [x for x in os.listdir(os.path.dirname(filen)) if re.search('_labels', x, re.IGNORECASE)]
                
                if not chan_label_file_ses:
                    if chan_label_filename:
                        chan_label_file=file_in.chnames_update(os.path.join(raw_file_path_sub, chan_label_filename[0]), bids_settings, write=False)
                        values = partition(chan_label_file)
                    else:
                        chan_label_file = header['chan_info']['ch_names']
                        values = partition(header['chan_info']['ch_names'])
                else:
                    chan_label_file=file_in.chnames_update(os.path.join(os.path.dirname(filen), chan_label_file_ses[0]), bids_settings, write=False)
                    values = partition(chan_label_file)
                
                eeg_chan_idx = [i for i, x in enumerate(values) if x[0] not in list(bids_settings['natus_info']['ChannelInfo'].keys())]
                group_info = determine_groups(np.array(chan_label_file)[eeg_chan_idx])
                
                file_info = [
                        ('FileName', ifile),
                        ('SubDir', raw_file_path_sub.split(os.path.sep)[-1]),
                        ('DisplayName', ifile if not sub_dir else ifile.split(os.path.sep)[0]),
                        ('Subject', header['meas_info']['subject_id']),
                        ('Gender', header['meas_info']['gender']),
                        ('Age', int(np.floor(((datetime.datetime.strptime(header['meas_info']['meas_date'].split(' ')[0],"%Y-%m-%d") 
                                - datetime.datetime.strptime(datetime.datetime.strptime(header['meas_info']['birthdate'],
                                '%d-%b-%Y').strftime('%Y-%m-%d'),"%Y-%m-%d")).days)/365)) if header['meas_info']['birthdate'] != 'X' else 'X'),
                        ('Birthdate', header['meas_info']['birthdate']),
                        ('RecordingID', header['meas_info']['recording_id']),
                        ('Date', header['meas_info']['meas_date'].split(' ')[0]),
                        ('Time', header['meas_info']['meas_date'].split(' ')[1]),
                        ('DataOffset', header['meas_info']['data_offset']),
                        ('NRecords', header['meas_info']['n_records']),
                        ('RecordLength', header['meas_info']['record_length']),
                        ('TotalRecordTime', round((((header['meas_info']['n_records']*(header['meas_info']['sampling_frequency']*header['meas_info']['record_length']))/header['meas_info']['sampling_frequency'])/60)/60,3)),
                        ('NChan', header['meas_info']['nchan']),
                        ('SamplingFrequency', header['meas_info']['sampling_frequency']),
                        ('Highpass', header['meas_info']['highpass']),
                        ('Lowpass', header['meas_info']['lowpass']),
                        ('Groups', group_info),
                        ('EDF_type', header['meas_info']['subtype']),
                        ('ses_chan_label', chan_label_file_ses),
                        ('chan_label', chan_label_filename)]
                
                file_info = OrderedDict(file_info)
                    
                if file_info['DisplayName'].lower().endswith('.edf'):
                    if file_info['NChan'] < 60:
                        file_info['RecordingType'] = 'Scalp'
                    else:
                        file_info['RecordingType'] = 'iEEG'
                    
                    if file_info['TotalRecordTime'] < 5:
                        file_info['RecordingLength'] = 'clip'
                    else:
                        file_info['RecordingLength'] = 'full'
                    
                    file_info['Retro_Pro'] = 'Pro'
                    
                else:
                    if 'ieeg' in file_info['DisplayName'].lower():
                        file_info['RecordingType'] = 'iEEG'
                    else:
                        file_info['RecordingType'] = 'Scalp'
                        
                    file_info['RecordingLength'] = file_info['DisplayName']
                    
                    if 'ret' in file_info['DisplayName'].lower():
                        file_info['Retro_Pro'] = 'Ret'
                    else:
                        file_info['Retro_Pro'] = 'Pro'
                        
                if file_info['RecordingType'] == 'Scalp':
                    chan_type = 'EEG'
                elif file_info['RecordingType'] == 'iEEG':
                    chan_type = 'SEEG'
                    
                ch_info = {}
                ch_info[chan_type] = OrderedDict([
                    ('ChannelCount', len(eeg_chan_idx)),
                    ('Unit', np.array(header['chan_info']['units'])[eeg_chan_idx]),
                    ('ChanName', np.array(chan_label_file)[eeg_chan_idx]),
                    ('Type', chan_type)])
                
                for key in bids_settings['natus_info']['ChannelInfo'].keys():
                    chan_idx = [i for i, x in enumerate(values) if x[0] == key]
                    ch_info[key] = OrderedDict([
                            ('ChannelCount', len(chan_idx)),
                            ('Unit', np.array(header['chan_info']['units'])[chan_idx]),
                            ('ChanName', np.array(chan_label_file)[chan_idx]),
                            ('Type', bids_settings['natus_info']['ChannelInfo'][key]['name'])])
                
                file_info['ChanInfo'] = OrderedDict(ch_info)
                file_info = OrderedDict(file_info)
            
                filesInfo_ses.append(file_info)
            except:
                print('Something wrong with file', ifile)
                raise
        
        if filesInfo_ses:
            filesInfo.append([filesInfo_ses, {'Date': filesInfo_ses[0]['Date'], 'Time': filesInfo_ses[0]['Time']}])
        
    filesInfo = sorted(filesInfo, key = lambda i: (i[1]['Date'], i[1]['Time']))
    filesInfo = [x[0] for x in filesInfo]
    
    if not sub_dir:
        filesInfo_final = []
        study_dates = [x[0]['Date'] for x in filesInfo]
        dates_unique, counts = np.unique(study_dates, return_counts=True)
        ses_cnt=1
        for idate in dates_unique:
            filesInfo_temp = []
            index_session = [i for i,x in enumerate(study_dates) if x == idate]
            for ises in index_session:
                filesInfo[ises][0]['session']=ses_cnt
                filesInfo_temp.append(filesInfo[ises][0])
            filesInfo_temp = sorted(filesInfo_temp, key = lambda i: (i['Date'], i['Time']))
            filesInfo_final.append(filesInfo_temp)
            ses_cnt+=1
            
        filesInfo = filesInfo_final
        
    return filesInfo

def folders_in(path_to_parent):
    contains_dir = False
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent,fname)):
            contains_dir = True
    
    return contains_dir

#%%

# input_dir = r'/media/veracrypt6/input'
# output_path = r'/media/veracrypt6/output'
# input_dir = r'/media/veracrypt6/projects/eplink/walkthrough_example/ieeg_sample/input'
# output_path = r'/media/veracrypt6/projects/eplink/walkthrough_example/ieeg_sample/output'
# input_dir = r'/media/veracrypt6/projects/iEEG/working_dir/edf2bids/input'
# output_path = r'/media/veracrypt6/projects/iEEG/working_dir/edf2bids/output'

# bids_settings = {}
# bids_settings['json_metadata'] = ieeg_file_metadata
# bids_settings['natus_info'] = natus_info
# bids_settings['settings_panel'] = {'Deidentify_source': False,
#                                      'offset_dates': False}
# file_info, chan_label_file, imaging_data = read_input_dir(input_dir, bids_settings)
# for isub in list(file_info):
#     file_info[isub]=sum(file_info[isub],[])
# new_sessions = read_output_dir(output_path, file_info, False, bids_settings, participants_fname=None)
# isub = list(file_info)[0]
# values = file_info[isub]
# offset_date = False


def read_input_dir(input_dir, bids_settings):
    """
    Reads data within the specified path.
    
    :param input_dir: Directory to read data files from.
    :type input_dir: string
    
    :return sub_file_info: Dictionary containing header information for all files in specified path.
    :type sub_file_info: dictionary
    
    """
    folders = [x for x in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, x))]
    folders = sorted_nicely([x for x in folders if len(os.listdir(os.path.join(input_dir,x))) !=0])
        
    sub_file_info = {}
    sub_chan_file = {}
    imaging_dir = {}
    for ifold in folders:
        subject_id = ifold.replace('_','')
        if not subject_id.startswith('sub-'):
            subject_id = 'sub-' + subject_id
            
        raw_file_path_sub = os.path.join(input_dir, ifold)
        file_info = get_file_info(raw_file_path_sub, bids_settings)

        sub_file_info[subject_id] = file_info
        chan_label_file = [x for x in os.listdir(raw_file_path_sub) if re.search('_labels', x, re.IGNORECASE)]
        imaging_data = [x for x in os.listdir(raw_file_path_sub) if 'imaging' in x.lower() and os.path.isdir(os.path.join(raw_file_path_sub,x))]
        sub_chan_file[subject_id] = chan_label_file
        imaging_dir[subject_id] = {}
        imaging_dir[subject_id]['imaging_dir'] = imaging_data
        imaging_dir[subject_id]['orig_sub_dir'] = ifold
        
    return sub_file_info, sub_chan_file, imaging_dir

def read_output_dir(output_path, file_info, offset_date, bids_settings, participants_fname):
    """
    Reads data within the specified output path.
    
    :param output_path: Output directory to read data files from.
    :type output_path: string
    :param file_info: File header information from input directory path.
    :type file_info: dictionary
    :param participants_fname: Filename for the participants tsv file.
    :type participants_fname: string
    
    :return new_sessions: Dictionary identifying if there are new sessions to process (comparing input to output).
    :type sub_file_info: dictionary
    
    """
    folders = sorted_nicely([x for x in os.listdir(output_path) if os.path.isdir(os.path.join(output_path,x)) and 'code' not in x])
    new_sessions = {}
        
    for isub, values in file_info.items():
        num_sessions = len(values)
        newSessions = True
        session_start = 0
        session_labels = np.nan
        sub_sessions = {}
        session_list = []
        all_labels = []
        unique_sessions = []
        
        # True if subject directory already in output directory
        if isub in folders:
            # Determine if new sessions to run, and where to begin
            if os.path.exists(os.path.join(output_path, isub, isub + '_scans.tsv')):
                # Load output scan file
                scans_tsv = pd.read_csv(os.path.join(output_path, isub, isub + '_scans.tsv'), sep='\t')
                sessionsDone = {}
                sessionsDone['date'] = [x for x in scans_tsv['acq_time']]
                sessionsDone['session'] = ['ses' + x.split('ses')[-1].split('_')[0] for x in scans_tsv['filename']]
                
                session_time = []
                session_dates = []
                session_labels = []
                if '_SE' not in values[0][0]['DisplayName']:
                    sess_cnt = [int(s) for s in re.findall(r'\d+', sessionsDone['session'][-1])][0]
                
                for ises in range(len(values)):
                    if offset_date:
                        date_study = datetime.datetime.strptime(values[ises][0]['Date'],"%Y-%m-%d")
                        date_study = (date_study - datetime.timedelta(5856)).strftime('%Y-%m-%d')
                    else:
                        date_study = values[ises][0]['Date']
                    
                    if '_SE' in values[ises][0]['DisplayName']:
                        if ''.join([date_study,'T', values[ises][0]['Time']]) not in scans_tsv['acq_time'].values:
                            visit_num = 'V'+ values[ises][0]['DisplayName'].split('_')[3]
                            session_num = values[ises][0]['DisplayName'].split('_')[4]
                            session_labels.append('ses-' + visit_num + session_num)
                            session_dates.append(date_study)
                            session_time.append(values[ises][0]['Time'])
                    else:
                        if ''.join([date_study,'T', values[ises][0]['Time']]) not in scans_tsv['acq_time'].values:
                            sess_cnt +=1
                            session_labels.append('ses-' + str(sess_cnt).zfill(3))
                            session_dates.append(date_study)
                            session_time.append(values[ises][0]['Time'])
                
                if session_dates:
                    sort_idx = pd.DataFrame({'date': [x.split('T')[0] for x in sessionsDone['date']] + session_dates,
                                                'time': [x.split('T')[1] for x in sessionsDone['date']] + session_time,
                                                'old_sess': sessionsDone['session'] + session_labels}).sort_values(by=['date','time']).index.values
            
                    session_combined = pd.DataFrame({'date': [[*[x.split('T')[0] for x in sessionsDone['date']], *session_dates][i] for i in sort_idx],
                                        'time': [[*[x.split('T')[1] for x in sessionsDone['date']], *session_time][i] for i in sort_idx],
                                        'old_sess':[*sessionsDone['session'], *session_labels],
                                        'new_sess': [[*sessionsDone['session'], *session_labels][i] for i in sort_idx]})
                    session_combined['date'] = session_combined['date'] + 'T' + session_combined['time']
                    del session_combined['time']
                        
                    unique_sessions = session_combined[(~session_combined.date.isin(scans_tsv['acq_time'].values))]
                    all_labels = list(session_combined['old_sess'].values)
                    session_labels = list(unique_sessions['new_sess'].values)
                    num_sessions = len(all_labels)
                    if len(values) == len(all_labels):
                        session_index = [x for x,i in enumerate(list(np.unique(all_labels))) if i in session_labels]
                    else:
                        session_index = [x for x,i in enumerate(session_labels) if i in list(np.unique(all_labels))]
                    session_start = len(set(sessionsDone['session']))
                    num_sessions = len(all_labels)
                    session_list = list(zip(list(session_combined['old_sess'].values), list(session_combined['new_sess'].values)))
                    
                else:
                    all_labels = sessionsDone['session']
                    session_labels = sessionsDone['session']
                    newSessions = False

    else:
        session_labels = []
        session_index = list(range(len(values)))
        for ises in range(len(values)):
            sess_temp=[]
            if '_SE' in values[ises]['DisplayName']:
                visit_num = 'V'+ values[ises]['DisplayName'].split('_')[3]
                session_num = values[ises]['DisplayName'].split('_')[4]
                sess_temp='ses-' + visit_num + session_num
            else:
                    sess_temp='ses-' + str(values[ises]['session']).zfill(3)
            
            session_labels.append(sess_temp)
                        
            all_labels = session_labels
            
            if participants_fname is not None:
                bids_helper = bidsHelper(subject_id=isub, output_path=output_path, bids_settings=bids_settings)
                bids_helper.write_participants(values[0][0])
            
        sub_sessions['newSessions'] = newSessions
        sub_sessions['session_done'] = session_start
        sub_sessions['num_sessions'] = num_sessions
        sub_sessions['session_labels'] = session_labels
        sub_sessions['session_index'] = session_index
        sub_sessions['all_sessions'] = all_labels
        sub_sessions['session_changes'] = session_list
        
        new_sessions[isub] = sub_sessions
        
    return new_sessions

def fix_sessions(session_list, num_sessions, output_path, isub):
    for ises in session_list:        
        old_fold = os.path.join(output_path, isub, ises[0], 'ieeg')
        new_fold = os.path.join(output_path, isub, 'temp', ises[1], 'ieeg')
        
        if not os.path.exists(os.path.join(output_path, isub, 'temp')):
            os.mkdir(os.path.join(output_path, isub, 'temp'))
        if not os.path.exists(os.path.join(output_path, isub, 'temp', ises[1])):
            os.mkdir(os.path.join(output_path, isub, 'temp', ises[1]))
        if not os.path.exists(new_fold):
            os.mkdir(new_fold)
            
        moveAllFilesinDir(old_fold, new_fold)
        
        for f in os.listdir(new_fold):
            if 'ieeg.edf' in f:
                new_filename = os.path.join(new_fold,'_'.join([f.split('_')[0],ises[1],f.split('_')[2], f.split('_')[3]]))
                os.rename(os.path.join(new_fold,f), new_filename)
                
                with open(new_filename, 'rb') as fid:
                    assert(fid.tell() == 0)
                    fid.seek(8)
                    fid.write(padtrim(isub, 80)) # subject id
                    fid.write(padtrim('Startdate X X X X', 80)) # recording id
                    
            elif 'ieeg.json' in f:
                new_filename = os.path.join(new_fold,'_'.join([f.split('_')[0],ises[1],f.split('_')[2], f.split('_')[3]]))
                os.rename(os.path.join(new_fold,f), new_filename)
            elif 'channels.tsv' in f:
                new_filename = os.path.join(new_fold,'_'.join([f.split('_')[0],ises[1],f.split('_')[2], f.split('_')[3]]))
                os.rename(os.path.join(new_fold,f), new_filename)
            elif 'events.tsv' in f:
                new_filename = os.path.join(new_fold,'_'.join([f.split('_')[0],ises[1],f.split('_')[2], f.split('_')[3]]))
                os.rename(os.path.join(new_fold,f), new_filename)
            elif 'electrodes.tsv' in f:
                new_filename = os.path.join(new_fold,'_'.join([f.split('_')[0],ises[1],f.split('_')[2]]))
                os.rename(os.path.join(new_fold,f), new_filename)
        
        shutil.rmtree(os.path.dirname(old_fold))
        
        old_code_fold = os.path.join(output_path, 'code', 'edf2bids', isub, ises[0])
        new_code_fold = os.path.join(output_path, 'code', 'edf2bids', isub, 'temp', ises[1])
        
        if not os.path.exists(os.path.join(output_path, 'code', 'edf2bids', isub, 'temp')):
            os.mkdir(os.path.join(output_path, 'code', 'edf2bids', isub, 'temp'))
        if not os.path.exists(new_code_fold):
            os.mkdir(new_code_fold)
        
        moveAllFilesinDir(old_code_fold, new_code_fold)
        shutil.rmtree(old_code_fold)
    
        # Load output scan file
        scans_tsv = pd.read_csv(os.path.join(output_path, isub, isub + '_scans.tsv'), sep='\t')
        name_idx = [i for i,x in enumerate(list(scans_tsv['filename'])) if ises[0] in x][-1]
        old_name = scans_tsv.loc[name_idx, 'filename'].split('/')[-1]
        new_name = 'ieeg/' + '_'.join([old_name.split('_')[0], ises[1]] + old_name.split('_')[2:])
        scans_tsv.loc[name_idx, 'filename'] = new_name
        
        bidsHelper._write_tsv(os.path.join(output_path, isub, isub + '_scans.tsv'), scans_tsv, float_form='%.3f', overwrite=True, verbose=False, append = False)

    temp_dir = os.path.join(output_path, isub, 'temp')
    final_dir = os.path.join(output_path, isub)
    
    moveAllFilesinDir(temp_dir, final_dir)
    
    shutil.rmtree(temp_dir)
    
    temp_dir = os.path.join(output_path, 'code', 'edf2bids', isub, 'temp')
    final_dir = os.path.join(output_path, 'code', 'edf2bids', isub)
    
    moveAllFilesinDir(temp_dir, final_dir)
    
    shutil.rmtree(temp_dir)
        
def moveAllFilesinDir(old_fold, new_fold):
    # Check if both the are directories
    if os.path.isdir(old_fold) and os.path.isdir(new_fold) :
        # Iterate over all the files in source directory
        for filePath in glob.glob(old_fold + os.path.sep + '*'):
            # Move each file to destination Directory
            shutil.move(filePath, new_fold)

def deidentify_edf_prv(source_fname,data_fname, isub, offset_date, rename):
    file_in = EDFReader()
    header = file_in.open(data_fname)
    
    edf_deidentify = {}
    edf_deidentify['subject_code'] = 'X' if not header['meas_info']['subject_code'] == 'X' else header['meas_info']['subject_code']
    edf_deidentify['birthdate'] = 'X' if not header['meas_info']['birthdate'] == 'X' else header['meas_info']['birthdate']
    edf_deidentify['gender'] = 'X' if not header['meas_info']['gender'] == 'X' else header['meas_info']['gender']
    edf_deidentify['subject_id'] = 'X' if not header['meas_info']['subject_id'] == 'X' else header['meas_info']['subject_id']
    recording_id = 'Startdate X X X X' if not header['meas_info']['recording_id'] == 'Startdate X X X X' else header['meas_info']['recording_id']
    
    new_date = []
    days_off = 0
    if offset_date:
        date_offset = datetime.datetime(2001,1,1)
        date_study = datetime.datetime.strptime(datetime.datetime.strptime(header['meas_info']['meas_date'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'), '%Y-%m-%d')
        days_offset = datetime.timedelta((date_study - date_offset).days - 1000)
        new_date = (date_offset + days_offset).strftime('%d.%m.%y')
        days_off = days_offset.days
    
    if (data_fname.lower().endswith(".edf")):
        with open(data_fname, 'r+b') as fid:
            assert(fid.tell() == 0)
            fid.seek(8)
            
            fid.write(padtrim('X X X X', 80))
            fid.write(padtrim(recording_id, 80))
            
            if new_date:
                fid.write(padtrim(new_date, 8))
            else:
                fid.seek(80+80+8+8)
                assert(fid.tell() == 80+80+8+8)

    elif (data_fname.lower().endswith(".edfz")) or (data_fname.lower().endswith(".edf.gz")):
        with open(data_fname, 'ab') as fid:
            assert(fid.tell() == 0)
            fid.seek(8)
            
            fid.write(padtrim('X X X X', 80))
            fid.write(padtrim(recording_id, 80))
            
            if new_date:
                fid.write(padtrim(new_date, 8))
            else:
                fid.seek(80+80+8+8)
                assert(fid.tell() == 80+80+8+8)
    
    if rename:
        if any(substring in data_fname.split(os.path.sep)[-1] for substring in {'~','_','-'}):
            new_name = data_fname.split(os.path.sep)[-1].replace(' ','').replace('~','_').replace('-','_').split('_')
            if len(new_name)>2:
                new_name = os.path.join(os.path.dirname(data_fname), '_'.join([isub, ''.join(new_name[2:])]))
                os.rename(data_fname, new_name)
        else:
            new_name = data_fname
    else:
        new_name = data_fname
    
    return new_name, days_off


def deidentify_edf(source_fname,data_fname, isub, offset_date, rename):
    file_in = EDFReader()
    header = file_in.open(data_fname)
    
    edf_deidentify = {}
    edf_deidentify['subject_code'] = 'X' if not header['meas_info']['subject_code'] == 'X' else header['meas_info']['subject_code']
    edf_deidentify['birthdate'] = 'X' if not header['meas_info']['birthdate'] == 'X' else header['meas_info']['birthdate']
    edf_deidentify['gender'] = 'X' if not header['meas_info']['gender'] == 'X' else header['meas_info']['gender']
    edf_deidentify['subject_id'] = 'X' if not header['meas_info']['subject_id'] == 'X' else header['meas_info']['subject_id']
    
    days_off = 0
    if offset_date:
        date_offset = datetime.datetime(2001,1,1)
        date_study = datetime.datetime.strptime(datetime.datetime.strptime(header['meas_info']['meas_date'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'), '%Y-%m-%d')
        days_offset = datetime.timedelta((date_study - date_offset).days - 1000)
        days_off = days_offset.days
    
   
    if rename:
        if any(substring in data_fname.split(os.path.sep)[-1] for substring in {'~','_','-'}):
            new_name = data_fname.split(os.path.sep)[-1].replace(' ','').replace('~','_').replace('-','_').split('_')
            if len(new_name)>2:
                new_name = os.path.join(os.path.dirname(data_fname), '_'.join([isub, ''.join(new_name[2:])]))
                os.rename(data_fname, new_name)
        else:
            new_name = data_fname
    else:
        new_name = data_fname
    
    header.close()
    
    return new_name, days_off

from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QMessageBox

# class warningBox(QMessageBox):
#     """Class for warning message box.

#     Displays a warning message box to the user.

#     Parameters
#     ----------
#     text: str
#         Text to appear in the error box.
#     """
#     def __init__(self, text):
#         super().__init__()
#         self.setIcon(QMessageBox.Icon.Critical)
#         self.setText(text)
#         self.setWindowTitle("Error")
#         self.exec()

from PyQt6.QtWidgets import QMessageBox


class warningBox(QMessageBox):
    """Class for warning message box.
    
    Displays a warning message box to the user.
    
    Parameters
    ----------
    text: str or list of str
    Text or list of strings to appear in the error box.
    """
    
    def __init__(self, text):
        super().__init__()
        self.setIcon(QMessageBox.Icon.Critical)
        self.setWindowTitle("Error")
        
        # Check if input is a list and join with newlines
        if isinstance(text, list):
            text = "\n".join(text)
        
        self.setText(text)
        self.exec()
