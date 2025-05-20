#!/usr/bin/env python3
import os
import re
import time
import shutil
import json
from datetime import datetime
from collections import OrderedDict

from edf_embedded_annotations_redactor import anonymize_edf_complete
from edf2bids import edf2bids
from helpers import EDFReader, determine_groups, bidsHelper

# ─── Configuration ─────────────────────────────────────────────────────────────

MAIN_FOLDER    = r'y:\_pipeline\Step_B_EDF_with_id_Monitored_PreBID'
DEST_FOLDER    = r'y:\_pipeline\Step_C_EDF_anon_repo'
ARCHIVE_FOLDER = r'y:\_pipeline\Step_B_ARCHIVE'
CHECK_INTERVAL = 10  # seconds
BIDS_SETTINGS  = "bids_settings.json"

# ─── No-op callback for edf2bids.write_annotations ──────────────────────────────

class DummySignal:
    def emit(self, *_): pass

# ─── Core Conversion Function ─────────────────────────────────────────────────

def convert_edf_to_bids(edf_path, dest_root, settings, session_index=1):
    """
    1) Anonymize+copy EDF
    2) Strip embedded annotations → events.tsv
    3) Build BIDS files (scans, sidecar, channels, electrodes)
    """
    # 1) Read header
    reader = EDFReader(edf_path)
    hdr    = reader.readHeader()
    subj   = hdr['meas_info']['subject_id']
    sess   = f"ses-{session_index:03d}"

    # 2) Make BIDS folders
    session_folder = os.path.join(dest_root, subj, sess)
    ieeg_folder    = os.path.join(session_folder, 'ieeg')
    os.makedirs(ieeg_folder, exist_ok=True)

    # 3) Anonymize & copy
    new_edf = f"{subj}_{sess}_ieeg.edf"
    dest_edf = os.path.join(ieeg_folder, new_edf)
    os.makedirs(os.path.dirname(dest_edf), exist_ok=True)
    anonymize_edf_complete(edf_path, dest_edf, log_dir=session_folder)

    # 4) Extract & write events.tsv
    worker = edf2bids()
    worker.annotation_fname = os.path.join(session_folder, f"{subj}_{sess}_events.tsv")
    # source=original, data_fname=anonymized copy, deidentify=True uses embedded redaction
    worker.write_annotations(
        source_fname=edf_path,
        data_fname=dest_edf,
        callback=DummySignal(),
        deidentify=True
    )

    # 5) Build scans record
    mi = hdr['meas_info']
    file_info_run = OrderedDict([
        ('filename', new_edf),
        ('acq_time', 'T'.join(mi['meas_date'].split(' '))),
        ('duration', round(mi['n_records'] * mi['record_length'] / 3600, 3)),
        ('edf_type', mi.get('subtype', 'EDF')),
    ])

    # 6) Instantiate bidsHelper & write out BIDS
    bh = bidsHelper(
        subject_id=subj,
        session_id=sess,
        task_id=None,
        run_num=None,
        kind='ieeg',
        suffix=None,
        output_path=os.path.join(dest_root, subj),
        bids_settings=settings,
        make_sub_dir=False
    )
    bh.write_dataset()
    bh.write_scans(new_edf, file_info_run, date_offset=False)
    bh.write_sidecar(file_info_run)
    bh.write_channels(file_info_run)
    bh.write_electrodes(file_info_run, coordinates=None)

    return session_folder

# ─── Utilities ────────────────────────────────────────────────────────────────

def is_file_in_use(path):
    try:
        tmp = path.replace(".edf", "_.edf")
        os.rename(path, tmp)
        os.rename(tmp, path)
        return False
    except:
        return True

def log_conversion(done_file, in_folder, out_folder, start, end):
    with open(done_file, 'w') as f:
        f.write(f"Conversion started:   {start}\n")
        f.write(f"Conversion completed: {end}\n")
        f.write(f"Input folder:  {in_folder}\n")
        f.write(f"Output folder: {out_folder}\n")

def move_to_archive(edf, edf_pass, subject):
    arch = os.path.join(ARCHIVE_FOLDER, subject)
    os.makedirs(arch, exist_ok=True)
    for p in (edf, edf_pass):
        try:
            shutil.move(p, os.path.join(arch, os.path.basename(p)))
        except Exception as e:
            print(f"⚠️ Archive failed for {p}: {e}")

# ─── Folder Watcher ───────────────────────────────────────────────────────────

def process_subject_folder(folder_name):
    in_f    = os.path.join(MAIN_FOLDER, folder_name)
    done_f  = os.path.join(in_f, "C_bids_conversion_done.txt")
    edfs    = [f for f in os.listdir(in_f) if f.lower().endswith(".edf")]
    if not edfs:
        return

    edf      = edfs[0]
    edf_p    = os.path.join(in_f, edf)
    edf_pass = edf_p + "_pass"
    print(f"▶️  Found EDF: {edf_p}")

    if not os.path.exists(edf_pass) or is_file_in_use(edf_p):
        return

    start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        settings = json.load(open(BIDS_SETTINGS))
        out_sess = convert_edf_to_bids(edf_p, DEST_FOLDER, settings, session_index=1)
        end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        log_conversion(done_f, in_f, out_sess, start, end)
        move_to_archive(edf_p, edf_pass, folder_name)
        print(f"✅ Done: {folder_name}")

    except Exception as e:
        print(f"❌ Error in {folder_name}: {e}")

def monitor_loop():
    print(f"👀 Monitoring {MAIN_FOLDER} every {CHECK_INTERVAL}s …")
    while True:
        for sub in sorted(os.listdir(MAIN_FOLDER)):
            path = os.path.join(MAIN_FOLDER, sub)
            if os.path.isdir(path) and re.match(r"sub-\d{3,}", sub):
                process_subject_folder(sub)
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    monitor_loop()
