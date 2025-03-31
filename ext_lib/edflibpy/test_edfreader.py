#!/usr/bin/env python3

import sys
import numpy as np
from edfreader import EDFreader

if sys.version_info[0] != 3 or sys.version_info[1] < 5:
  print("Must be using Python version >= 3.5.0")
  sys.exit()

if np.__version__ < "1.17.0":
  print("Must be using NumPy version >= 1.17.0")
  sys.exit()

if len(sys.argv) != 2:
  print("usage: test_edflib.py <filename>\n")
  sys.exit()

hdl = EDFreader(sys.argv[1])

print("\nStartdate: %02d-%02d-%04d" %(hdl.getStartDateDay(), hdl.getStartDateMonth(), hdl.getStartDateYear()))
print("Starttime: %02d:%02d:%02d" %(hdl.getStartTimeHour(), hdl.getStartTimeMinute(), hdl.getStartTimeSecond()))
filetype = hdl.getFileType()
if (filetype == hdl.EDFLIB_FILETYPE_EDF) or (filetype == hdl.EDFLIB_FILETYPE_BDF):
  print("Patient: %s" %(hdl.getPatient()))
  print("Recording: %s" %(hdl.getRecording()))
else:
  print("Patient code: %s" %(hdl.getPatientCode()))
  print("Gender: %s" %(hdl.getPatientGender()))
  print("Birthdate: %s" %(hdl.getPatientBirthDate()))
  print("Patient name: %s" %(hdl.getPatientName()))
  print("Patient additional: %s" %(hdl.getPatientAdditional()))
  print("Admin. code: %s" %(hdl.getAdministrationCode()))
  print("Technician: %s" %(hdl.getTechnician()))
  print("Equipment: %s" %(hdl.getEquipment()))
  print("Recording additional: %s" %(hdl.getRecordingAdditional()))
print("Reserved: %s" %(hdl.getReserved()))
edfsignals = hdl.getNumSignals()
print("Number of signals: %d" %(hdl.getNumSignals()))
print("Number of datarecords: %d" %(hdl.getNumDataRecords()))
print("Datarecord duration: %f" %(hdl.getLongDataRecordDuration() / 10000000.0))

n = edfsignals
if n > 3:
  n = 3

for i in range(0, n):
  print("\nSignal: %s" %(hdl.getSignalLabel(i)))
  print("Samplefrequency: %f Hz" %(hdl.getSampleFrequency(i)))
  print("Transducer: %s" %(hdl.getTransducer(i)))
  print("Physical dimension: %s" %(hdl.getPhysicalDimension(i)))
  print("Physical minimum: %f" %(hdl.getPhysicalMinimum(i)))
  print("Physical maximum: %f" %(hdl.getPhysicalMaximum(i)))
  print("Digital minimum: %d" %(hdl.getDigitalMinimum(i)))
  print("Digital maximum: %d" %(hdl.getDigitalMaximum(i)))
  print("Prefilter: %s" %(hdl.getPreFilter(i)))
  print("Samples per datarecord: %d" %(hdl.getSampelsPerDataRecord(i)))
  print("Total samples in file: %d" %(hdl.getTotalSamples(i)))
  print("Reserved: %s" %(hdl.getSignalReserved(i)))


n = len(hdl.annotationslist)

print("\nannotations in file: %d" %(n))

if n > 10:
  n = 10

for i in range(0, n):
  print("annotation: onset: %d:%02d:%02.3f    description: %s    duration: %d" %(\
        (hdl.annotationslist[i].onset / 10000000) / 3600, \
        ((hdl.annotationslist[i].onset / 10000000) % 3600) / 60, \
        (hdl.annotationslist[i].onset / 10000000) % 60, \
        hdl.annotationslist[i].description, \
        hdl.annotationslist[i].duration))

ibuf = np.empty(100, dtype = np.int32)
dbuf = np.empty(100, dtype = np.float_)

hdl.rewind(0)
hdl.readSamples(0, ibuf, 100)
hdl.rewind(0)
hdl.readSamples(0, dbuf, 100)

hdl.close()


for i in range(0, 100):
  print("buf[% 3d]:   %+8d   %+ 9.3f" %(i, ibuf[i], dbuf[i]))



















