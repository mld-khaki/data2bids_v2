#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 20:51:22 2020

@author: greydon
"""
import numpy as np
import re

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



chan_label_filename = r'/media/veracrypt6/projects/eplink/walkthrough_example/EPL31_LHS_0008_channel_labels.txt'
fname=r'/media/veracrypt6/projects/iEEG/ieeg/out/sub-023_ses-008_task-stim_run-02_ieeg.edf'

with open(fname, 'r+b') as fid:
	fid.seek(252)
	nchan = int(fid.read(4).decode())
	channels = list(range(nchan))
	ch_names_orig= [fid.read(16).strip().decode() for ch in channels]
	
chan_idx = [i for i, x in enumerate(ch_names_orig) if not any(x.startswith(substring) for substring in list(ChannelInfo.keys()))]

if chan_label_new.shape[1] >1:
	chan_label_new=[x[1] for x in chan_label_new]

if len(chan_label_new)!=len(chan_idx):
	replace_chan = [str(x) for x in list(range(len(chan_label_new)+1,len(chan_idx)+1))]
	chan_label_new.extend([''.join(list(item)) for item in list(zip(['C']*len(replace_chan), replace_chan))])
	assert len(chan_label_new)==len(chan_idx)

ch_names_new=ch_names_orig
for (index, replacement) in zip(chan_idx, chan_label_new):
	ch_names_new[index] = replacement

with open(fname, 'r+b') as fid:
	assert(fid.tell() == 0)
	fid.seek(256)
	for ch in ch_names_new:
		fid.write(padtrim(ch,16))




