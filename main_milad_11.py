# My routine using the EDF2BID


import sys
import os
import pandas as pd
import datetime
import shutil
import numpy as np
import json
from helpers import read_input_dir, read_output_dir, bidsHelper, fix_sessions
from edf2bids import edf2bids

cur_path = 'e:\\data2bids-master-v2p2\\'
sys.path.append(os.path.abspath(cur_path))


class Data2Bids:
    def __init__(self):
        self.application_path = cur_path
        self.settings_fname = os.path.join(self.application_path, 'bids_settings.json')
        self.version_fname = os.path.join(self.application_path, 'version.json')
        
        with open(self.version_fname) as version_file:
            self.app_info = json.load(version_file)
        
        self.settings = self.load_settings()
        self.input_path = None
        self.output_path = None
        self.file_info = {}
        self.chan_label_file = {}
        self.imaging_data = {}
        self.new_sessions = {}
        self.starting_session = 11
        
        # Conversion settings
        self.deidentify_source = False
        self.deidentify_tsv = False  # New setting for TSV anonymization
        self.gzip_edf = False
        self.offset_date = False
        self.dry_run = False

    def load_settings(self):
        """Load or create default settings"""
        if os.path.exists(self.settings_fname):
            with open(self.settings_fname) as settings_file:
                return json.load(settings_file)
                
        default_settings = {
            'general': {
                'recordingLabels': "full,clip,stim,ccep"
            },
            'json_metadata': {
                'TaskName': 'EEG Clinical',
                'Experimenter': ['John Smith'],
                'Lab': 'Some cool lab',
                'InstitutionName': 'Some University',
                'InstitutionAddress': '123 Fake Street, Fake town, Fake country',
                'ExperimentDescription': '',
                'DatasetName': '',
            },
            'natus_info': {
                'Manufacturer': 'Natus',
                'ManufacturersModelName': 'Neuroworks',
                'SamplingFrequency': 1000,
                'HighpassFilter': np.nan,
                'LowpassFilter': np.nan,
                'MERUnit': 'uV',
                'PowerLineFrequency': 60,
                'RecordingType': 'continuous',
                'iEEGCoordinateSystem': 'continuous',
                'iEEGElectrodeInfo': {
                    'Manufacturer': 'AdTech',
                    'Type': 'depth',
                    'Material': 'Platinum',
                    'Diameter': 0.86
                },
                'EEGElectrodeInfo': {
                    'Manufacturer': 'AdTech',
                    'Type': 'scalp',
                    'Material': 'Platinum',
                    'Diameter': 10
                }
            },
            'settings_panel': {
                'Deidentify_source': False,
                'Deidentify_tsv': False,  # New setting
                'offset_dates': False,
                'gzip_edf': True
            }
        }
        
        with open(self.settings_fname, 'w') as f:
            json.dump(default_settings, f, indent=4)
        
        return default_settings

    def anonymize_tsv(self, tsv_path):
        """Anonymize TSV files by removing/replacing sensitive information"""
        if os.path.exists(tsv_path):
            df = pd.read_csv(tsv_path, sep='\t')
            
            # Define columns that might contain sensitive information
            sensitive_cols = ['name', 'patient', 'subject', 'id', 'identifier', 'birth']
            
            # Replace sensitive information in column names and values
            for col in df.columns:
                if any(s in col.lower() for s in sensitive_cols):
                    if 'birth' in col.lower():
                        df[col] = 'XXXX-XX-XX'
                    # else:
                    #     df[col] = df[col].apply(lambda x: 'XXXXXXXXX' if pd.notnull(x) else x)
            
            # Save anonymized TSV
            df.to_csv(tsv_path, sep='\t', index=False)

    def setup_directories(self, input_dir, output_dir, ses_num=1):
        """Setup input and output directories and initialize data structures"""
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory does not exist: {input_dir}")
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        elif len(os.listdir(output_dir)) > 0:
            raise ValueError(f"Output directory is not empty: {output_dir}")
            
        self.input_path = input_dir
        self.output_path = output_dir
        
        # Load and process input directory
        print("Loading input directory...")
        self.file_info, self.chan_label_file, self.imaging_data = read_input_dir(self.input_path, self.settings)
        
        if not self.file_info:
            raise ValueError("No valid EDF files found in input directory")
            
        # Initialize sessions for each subject
        for subject_id, sessions in self.file_info.items():
            if isinstance(sessions, list) and sessions and isinstance(sessions[0], list):
                self.file_info[subject_id] = [item for sublist in sessions for item in sublist]

        # Setup BIDS directory structure
        print("Setting up BIDS directory structure...")
        self._setup_bids_directory()
        
        # Initialize new sessions structure
        self.new_sessions = read_output_dir(
            self.output_path,
            self.file_info,
            self.offset_date,
            self.settings,
            participants_fname=None,
            session_offset=ses_num
        )


    def convert(self,start_num=[]):
        """Run the actual conversion process"""
        if not self.file_info:
            raise ValueError("No files loaded. Run setup_directories() first.")
            
        print("\nStarting EDF to BIDS conversion...")
        
        # Initialize converter
        edf_converter = edf2bids()
        edf_converter.bids_settings = self.settings
        edf_converter.new_sessions = self.new_sessions
        edf_converter.file_info = self.file_info
        edf_converter.chan_label_file = self.chan_label_file
        edf_converter.input_path = self.input_path
        edf_converter.output_path = self.output_path
        edf_converter.script_path = self.application_path
        edf_converter.coordinates = None
        edf_converter.electrode_imp = None
        edf_converter.make_dir = True
        edf_converter.overwrite = True
        edf_converter.deidentify_source = self.deidentify_source
        edf_converter.gzip_edf = self.gzip_edf
        edf_converter.offset_date = self.offset_date
        edf_converter.dry_run = self.dry_run
        
        print("Converting files...")
        edf_converter.run()

        # Anonymize TSV files if enabled
        if self.deidentify_tsv and not self.dry_run:
            print("Anonymizing TSV files...")
            for root, _, files in os.walk(self.output_path):
                for file in files:
                    if file.endswith('.tsv'):
                        tsv_path = os.path.join(root, file)
                        self.anonymize_tsv(tsv_path)
        
        print("\nBIDS conversion complete!")

    def _setup_bids_directory(self):
        """Setup initial BIDS directory structure"""
        bids_helper = bidsHelper(output_path=self.output_path, bids_settings=self.settings)
        bids_helper.write_dataset()
        bids_helper.write_participants()
        
        for fname in ['bidsignore', 'README']:
            src = os.path.join(self.application_path, 'static', fname)
            dst = os.path.join(self.output_path, '.' + fname if fname == 'bidsignore' else fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy(src, dst)

def check_input_dir(input_dir):
    """Add this function to debug the issue"""
    files = os.listdir(input_dir)
    print(f"Files found: {len(files)}")
    for f in files[:5]:  # Print first 5 files
        full_path = os.path.join(input_dir, f)
        print(f"File: {f}")
        print(f"Exists: {os.path.exists(full_path)}")
        print(f"Readable: {os.access(full_path, os.R_OK)}")

# Configure paths
input_dir  = "e:/sub-030_PachecoRobert/"  # Replace with actual path
output_dir = "e:/sub-030_PachecoRobert/"  # Replace with actual path

check_input_dir(input_dir)



# Initialize converter
converter = Data2Bids()

# Set conversion options
converter.deidentify_source = False
converter.deidentify_tsv = True  # Enable TSV anonymization
converter.gzip_edf = False
converter.offset_date = False
converter.dry_run = False

session_offset = 1 # used for debugging, just keep it as one

try:
    # Setup directories and initialize sessions
    converter.setup_directories(input_dir, output_dir, session_offset)
    
    # Debug print to check structure
    print("\nDetected sessions:")
    for subject, info in converter.new_sessions.items():
        print(f"\nSubject {subject}:")
        print(f"  Number of sessions: {info['num_sessions']}")
        print(f"  Session labels: {info['session_labels']}")
        
    # Run conversion
    converter.convert()
    
    print("\nConversion completed successfully!")
    
except Exception as e:
    print(f"\nError during conversion: {str(e)}")
    raise(e)

