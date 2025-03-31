#!/usr/bin/env python3

import sys
import numpy as np
from edfwriter import EDFwriter

if sys.version_info[0] != 3 or sys.version_info[1] < 5:
  print("Must be using Python version >= 3.5.0")
  sys.exit()

if np.__version__ < "1.17.0":
  print("Must be using NumPy version >= 1.17.0")
  sys.exit()

sf1 = 1000  # samplerate signal 1
sf2 = 1500  # samplerate signal 2
edfsignals = 2
datrecs = 20  # recording length 20 seconds
sine_17 = np.pi * 2.0
sine_17 /= (sf1 / 1.7)
sine_26 = np.pi * 2.0
sine_26 /= (sf2 / 2.6)
q1 = 0.0
q2 = 0.0

hdl = EDFwriter("sinewave.edf", EDFwriter.EDFLIB_FILETYPE_EDFPLUS, edfsignals)

for chan in range(0, edfsignals):
  if hdl.setPhysicalMaximum(chan, 3000) != 0:
    print("setPhysicalMaximum() returned an error")
    sys.exit()
  if hdl.setPhysicalMinimum(chan, -3000) != 0:
    print("setPhysicalMinimum() returned an error")
    sys.exit()
  if hdl.setDigitalMaximum(chan, 32767) != 0:
    print("setDigitalMaximum() returned an error")
    sys.exit()
  if hdl.setDigitalMinimum(chan, -32768) != 0:
    print("setDigitalMinimum() returned an error")
    sys.exit()
  if hdl.setPhysicalDimension(chan, "uV") != 0:
    print("setPhysicalDimension() returned an error")
    sys.exit()
if hdl.setSampleFrequency(0, sf1) != 0:
  print("setSampleFrequency() returned an error")
  sys.exit()
if hdl.setSampleFrequency(1, sf2) != 0:
  print("setSampleFrequency() returned an error")
  sys.exit()
if hdl.setSignalLabel(0, "sine 1.7Hz") != 0:
  print("setSignalLabel() returned an error")
  sys.exit()
if hdl.setSignalLabel(1, "sine 2.6Hz") != 0:
  print("setSignalLabel() returned an error")
  sys.exit()

if hdl.setPreFilter(0, "HP:0.05Hz LP:250Hz N:60Hz") != 0:
  print("setPreFilter() returned an error")
  sys.exit()
if hdl.setPreFilter(1, "HP:0.05Hz LP:250Hz N:60Hz") != 0:
  print("setPreFilter() returned an error")
  sys.exit()
if hdl.setTransducer(0, "AgAgCl cup electrode") != 0:
  print("setTransducer() returned an error")
  sys.exit()
if hdl.setTransducer(1, "AgAgCl cup electrode") != 0:
  print("setTransducer() returned an error")
  sys.exit()

if hdl.setPatientCode("1234567890") != 0:
  print("setPatientCode() returned an error")
  sys.exit()
if hdl.setPatientBirthDate(1913, 4, 7) != 0:
  print("setPatientBirthDate() returned an error")
  sys.exit()
if hdl.setPatientName("Smith J.") != 0:
  print("setPatientName() returned an error")
  sys.exit()
if hdl.setAdditionalPatientInfo("normal condition") != 0:
  print("setAdditionalPatientInfo() returned an error")
  sys.exit()
if hdl.setAdministrationCode("1234567890") != 0:
  print("setAdministrationCode() returned an error")
  sys.exit()
if hdl.setTechnician("Black Jack") != 0:
  print("setTechnician() returned an error")
  sys.exit()
if hdl.setEquipment("recorder") != 0:
  print("setEquipment() returned an error")
  sys.exit()
if hdl.setAdditionalRecordingInfo("nothing special") != 0:
  print("setAdditionalRecordingInfo() returned an error")
  sys.exit()

buf1 = np.empty(sf1, np.float64, "C")
buf2 = np.empty(sf2, np.float64, "C")

for i in range(0, datrecs):
  for j in range(0, sf1):
    q1 += sine_17
    buf1[j] = 2000.0 * np.sin(q1)

  err = hdl.writeSamples(buf1)
  if err != 0:
    print("writeSamples() returned error: %d" %(err))
    break;

  for j in range(0, sf2):
    q2 += sine_26
    buf2[j] = 2000.0 * np.sin(q2)

  err = hdl.writeSamples(buf2)
  if err != 0:
    print("writeSamples() returned error: %d" %(err))
    break;

if hdl.writeAnnotation(0, -1, "Recording starts") != 0:
  print("writeAnnotation() returned an error")
if hdl.writeAnnotation(40000, 20000, "Test annotation") != 0:
  print("writeAnnotation() returned an error")

hdl.close()






















