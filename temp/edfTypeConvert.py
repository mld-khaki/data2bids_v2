# -*- coding: utf-8 -*-
"""

@author: Greydon
"""

import os
import subprocess
import numpy as np
import argparse

def edfC2D(file):
	with open(file, 'r+b') as fid:
		assert(fid.tell() == 0)
		fid.seek(192)
		fid.write(bytes(str("EDF+D") + ' ' * (44-len("EDF+D")), encoding="ascii"))

def edfD2C(file):
	with open(file, 'r+b') as fid:
		assert(fid.tell() == 0)
		fid.seek(192)
		fid.write(bytes(str("EDF+C") + ' ' * (44-len("EDF+C")), encoding="ascii"))


def main(args):
	if args.type == 'c':
		edfD2C(args.edf_file)
	elif args.type == 'd':
		edfC2D(args.edf_file)
	else:
		print("Please choose either 'c' or 'd' as type")

if __name__ == "__main__":
	
	# Input arguments
	parser = argparse.ArgumentParser(description="Converts EDF files to specified type (EDF+D or EDF+C).")
	parser.add_argument("-f", "--edf_file", dest="edf_file", help="Path to EDF file.")
	parser.add_argument("-t", "--type", dest="type", help="Desired EDF file type.")
	args = parser.parse_args()
	
	main(args)
