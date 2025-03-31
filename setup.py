# -*- coding: utf-8 -*-

# A simple setup script to create an executable using PyQt5. This also
# demonstrates the method for creating a Windows executable that does not have
# an associated console.
#
# PyQt5app.py is a very simple type of PyQt5 application
#
# Run the build process by running the command 'python setup.py build'
#
# If everything works well you should find a subdirectory in the build
# subdirectory that contains the files needed to run the application
#
import sys
from cx_Freeze import setup, Executable
sys.setrecursionlimit(5000)

base = 'Win32GUI' if sys.platform=='win32' else None

includepackages= ['google','googleapiclient','pydicom']
includefiles = ['edf2bids.py', 'helpers.py', 'bids2spred.py', 'dicom2bids.py', 'version.json', 'widgets/', 'static/', 'ext_lib/']
#includefiles+=['C:\\Users\\xluser\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\google_api_core-2.15.0.dist-info']
#includefiles+=['C:\\Users\\xluser\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\google']
#includefiles+=['C:\\Users\\xluser\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\python_gdcm-3.0.22.dist-info']

buildOptions = {
		'build_exe': {
		'includes': 'atexit',
		"include_files": includefiles,
		"packages": includepackages,
		'include_msvcr': True
		}
}

executables = [
	Executable('main.py', base=base, target_name = 'data2bids', icon="static/eplink_icon.ico")
]

setup(name='data2bids',
		version = '1.5',
		description = 'Convert clinical data files to BIDS/SPReD format.',
		options = buildOptions,
		executables = executables)
