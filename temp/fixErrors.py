# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 19:12:15 2019

@author: Greydon
"""
import os
import json
import re
import gzip
from edf2bids.helpers import EDFReader, is_float
import mne
import pandas as pd
import numpy as np
from collections import OrderedDict
import shutil

def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

ignoreNotes = ('Closed','De-block','Opened','Video','Impedance', 'Stop Recording','Saturation Recovery','Breakout box','Gain/Filter','Clip Note',
                   'Headbox Disconnecting', 'Headbox Reconnected')

data_dir = r'D:\projects\iEEG\code\edf2bids'
scene_dir = r'F:\projects\iEEG\scenes'
script_path = r'C:\Users\Greydon\Documents\github\edf2bids\edf2bids'

subs = [x for x in os.listdir(data_dir) if x.startswith('sub')]
isub = subs[0]
ises = sessions[0]
for isub in subs:
    sessions = sorted_nicely([x for x in os.listdir(os.path.join(data_dir, isub)) if x.startswith('ses')])
    
    for ises in sessions:
        code_path = os.path.join(data_dir, isub, ises)
        
        os.remove(os.path.join(code_path,'edf2bids.py'))
        shutil.copy(os.path.join(script_path, 'edf2bids.py'), code_path)
    
#        coords = [x for x in os.listdir(os.path.join(data_dir, isub, ises, 'ieeg')) if x.endswith('coordsystem.json')][0]
#        
#        fname = os.path.join(data_dir, isub, ises, 'ieeg', coords)
#        with open(fname) as _file:
#            coord_json = json.load(_file)
#        fileN = [x for x in os.listdir(os.path.join(data_dir, isub, ises, 'ieeg')) if '_ieeg.edf' in x]
#        coord_json['IntendedFor'] = "/".join(coord_json['IntendedFor'].split('/')[:-1] + fileN)
#        
#        json_output = json.dumps(coord_json, indent=4)
#        with open(fname, 'w') as fid:
#            fid.write(json_output)
#            fid.write('\n')
        
        #----------------------------------------------------------#
        #                         CHANNELS tsv                     #
        #----------------------------------------------------------#
#        chans = [x for x in os.listdir(os.path.join(data_dir, isub, ises, 'ieeg')) if x.endswith('channels.tsv')][0]
#        
#        fname = os.path.join(data_dir, isub, ises, 'ieeg', chans)
#        chans_table = pd.read_table(fname)
#        if 'sampling_frequency' not in list(chans_table):
#            chans_table['sampling_frequency'] = np.repeat(1000, len(chans_table['name']))
#        
#        chans_table = chans_table[['name','type','units','low_cutoff','high_cutoff','notch','reference','group','sampling_frequency']]
#        chans_table.to_csv(fname, sep='\t', index=False, na_rep='n/a')
        
        #----------------------------------------------------------#
        #                        ELECTRODES tsv                    #
        #----------------------------------------------------------#
#        elects = [x for x in os.listdir(os.path.join(data_dir, isub, ises, 'ieeg')) if x.endswith('electrodes.tsv')][0]
#        
#        fname = os.path.join(data_dir, isub, ises, 'ieeg', elects)
#        elects_table = pd.read_table(fname)
##        if 'type' not in list(elects_table):
##            elects_table['type'] = np.repeat('depth', len(elects_table['name']))
#
#        elects_table = elects_table[['name','x','y','z']]
#        elects_table.to_csv(fname, sep='\t', index=False, na_rep='n/a')
        
        #----------------------------------------------------------#
        #                    Extract Coordinates                   #
        #----------------------------------------------------------#
        elects = [x for x in os.listdir(os.path.join(data_dir, isub, ises, 'ieeg')) if x.endswith('electrodes.tsv')][0]
        
        
        fname = os.path.join(data_dir, isub, ises, 'ieeg', elects)
        elects_table = pd.read_table(fname)
        if 'type' not in list(elects_table):
            elects_table['type'] = np.repeat('depth', len(elects_table['name']))
        
        if 'size' not in list(elects_table):
            elects_table['size'] = np.repeat(1.2, len(elects_table['name']))
            
        elects_table = elects_table[['name','x','y','z','size','type']]
        elects_table.to_csv(fname, sep='\t', index=False, na_rep='n/a')
        
        #----------------------------------------------------------#
        #                          ANNOTATIONS                     #
        #----------------------------------------------------------#
        annot = [x for x in os.listdir(os.path.join(data_dir, isub, ises, 'ieeg')) if x.endswith('annotations.tsv')]
        if not annot:
            fileN = [x for x in os.listdir(os.path.join(data_dir, isub, ises, 'ieeg')) if x.endswith('_ieeg.edf.gz')]
            if fileN:
                delete_temp = True
                file_temp = os.path.join(data_dir, isub, ises, 'ieeg', fileN[0]).split('.gz')[0]
                with gzip.open(os.path.join(data_dir, isub, ises, 'ieeg', fileN[0]), 'rb') as f_in:
                    with open(file_temp, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                delete_temp = False
                
            fileN = [x for x in os.listdir(os.path.join(data_dir, isub, ises, 'ieeg')) if x.endswith('_ieeg.edf')][0]
            
            file_in = EDFReader()
            file_in.open(os.path.join(data_dir, isub, ises, 'ieeg',fileN))
            header = file_in.readHeader()
            
            chanindx = [x for x in header[1]['ch_names'] if not x.endswith('Annotations')]
            annotTemp = mne.io.read_raw_edf(os.path.join(data_dir, isub, ises, 'ieeg',fileN), preload=True, exclude = chanindx[1:], verbose=None)
            events_edf = annotTemp.find_edf_events()[[i for i,x in enumerate(annotTemp.find_edf_events()) if not x[2].isdigit() and not is_float(x[2]) and not x[2].startswith(ignoreNotes)]]
            file_in.close()
    
            df = pd.DataFrame(OrderedDict([
                              ('timestamp', [x[0] for x in events_edf]),
                              ('event', [x[2] for x in events_edf])]))
            fname = os.path.join(data_dir, isub, ises, 'ieeg', '_'.join([isub,ises,'run-01','annotations.tsv']))
            df.to_csv(fname, sep='\t', index=False, na_rep='n/a')
            
            if delete_temp:
                os.remove(file_temp)
            
            print(('Finished annotations file for subject {} session {}').format(isub, ises))
        
        #----------------------------------------------------------#
        #                       GZIP edf files                     #
        #----------------------------------------------------------#
        fileN = [x for x in os.listdir(os.path.join(data_dir, isub, ises, 'ieeg')) if x.endswith('_ieeg.edf')]
        if fileN:
            raw_file_path = os.path.join(data_dir, isub, ises, 'ieeg',fileN[0])

            with open(raw_file_path, 'rb') as src, gzip.open(raw_file_path + '.gz', 'wb') as dst:        
                    dst.writelines(src)
#            with open(raw_file_path, 'rb') as src, gzip.open(raw_file_path + '.gz', 'wb') as dst:
#                shutil.copyfileobj(src, dst)
                
            os.remove(raw_file_path)
            
            print(('Finished subject {} session {}').format(isub, ises))
        
data_dir = r'F:\projects\iEEG\bids'
subs = [x for x in os.listdir(data_dir) if x.startswith('sub')]
isub = subs[2]
for isub in subs:
    filen = os.path.join(data_dir, isub, isub + '_scans.tsv')
    scans_tsv = pd.read_table(filen)
    scans = scans_tsv['filename']
#    iscan = scans[13]
    scan_dur = []
    for iscan in scans:
        ses = 'ses-' + iscan.split('ses-')[-1].split('_')[0]
        if iscan.endswith('.gz'):
            delete_temp = True
            file_temp = os.path.join(data_dir, isub, ses, iscan.split('.gz')[0])
            with gzip.open(os.path.join(data_dir, isub, ses, iscan), 'rb') as f_in:
                with open(file_temp, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            delete_temp = False
            file_temp = os.path.join(data_dir, isub, ses, iscan)
            
        file_in = EDFReader()
        meas_info, chan_info = file_in.open(file_temp)
        fname = os.path.join(data_dir, isub, ses, iscan.split('.edf')[0] + '.json')
        with open(fname) as _file:
            eeg_json = json.load(_file)
        
        eeg_json['RecordingDuration'] = round((((meas_info['n_records']*128)/1000)/60)/60,3)
        json_output = json.dumps(eeg_json, indent=4)
        with open(fname, 'w') as fid:
            fid.write(json_output)
            fid.write('\n')
        
        if delete_temp:
            os.remove(file_temp)
                
        scan_dur.append(eeg_json['RecordingDuration'])
    
    scans_tsv['duration'] = pd.DataFrame(scan_dur)
    scans_tsv.to_csv(filen, sep='\t', index=False, na_rep='n/a')
    
    
    
    ACPC = [x for x in os.listdir(os.path.join(scene_dir, isub, 'scene')) if x.startswith('acpc')][0]
    acpcFile = pd.read_csv(os.path.join(scene_dir, isub, 'scene', ACPC),header=2,delimiter=',')
    if len(acpcFile)<3:
        mcpLat  = acpcFile.iloc[1,1] + (acpcFile.iloc[0,1] - acpcFile.iloc[1,1])/2
        mcpAP   = acpcFile.iloc[1,2] + (acpcFile.iloc[0,2] - acpcFile.iloc[1,2])/2
        mcpVert = acpcFile.iloc[1,3] + (acpcFile.iloc[0,3] - acpcFile.iloc[1,3])/2
        file = os.path.join(scene_dir, isub, 'scene', ACPC)
        fd = open(file,'a')
        fmt = "%s,%5.4f,%5.4f,%5.4f,%i,%i,%i,%i,%i,%i,%i,%s,%s,%s\n"
        node = ('vtkMRMLMarkupsFiducialNode_' + "2")
        fd.write(fmt % (node, mcpLat,mcpAP,mcpVert,0,0,0,1,1,1,1,'MCP','','vtkMRMLScalarVolumeNode4'))
        fd.close()
    
    acpcFile = pd.read_csv(os.path.join(scene_dir, isub, 'scene', ACPC),header=2,delimiter=',')
    mcp = acpcFile.iloc[2,1:4]
    fids = [x for x in os.listdir(os.path.join(scene_dir, isub, 'scene')) if x.startswith('fiducials')][0]
    fidsFile = pd.read_csv(os.path.join(scene_dir, isub, 'scene', fids),header=2,delimiter=',')
    
    values = []
    for item in fidsFile['label']:
        item = item.replace('-','')
        if len(re.findall(r"([a-zA-Z]+)([0-9]+)",item))>1:
            first = "".join(list(re.findall(r"([a-zA-Z]+)([0-9]+)([a-zA-Z]+)",item)[0]))
            second = list(re.findall(r"([a-zA-Z]+)([0-9]+)",item)[-1])[-1]
            values.append([first, second])
        else:
            values.append(list(re.findall(r"([a-zA-Z]+)([0-9]+)",item)[0]))
    
    fidsFile['group'] = [x[0] for x in values]
    fidsFile['contact'] = [x[1] for x in values]
    fidsFile['x'] = round(fidsFile['x']-mcp[0],3)
    fidsFile['y'] = round(fidsFile['y']-mcp[1],3)
    fidsFile['z'] = round(fidsFile['z']-mcp[2],3)
    fidsFile = fidsFile[['group','contact','x','y','z']]
    
    fname = os.path.join(data_dir, isub,"_".join([isub, 'actual_coordinates.tsv']))
    fidsFile.to_csv(fname, sep='\t', index=False, na_rep='n/a')
    
        
    