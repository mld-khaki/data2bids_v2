# Changelog

All notable changes to this project will be documented in this file.

## [08.13.2021] (2021-08-13)

### Features

* add EDF convert type tool in the Tools menu - will overwrite the EDF type (EDF+C/EDF+D) specified in the EDF file header
	* occasionally, Natus will export EDF files incorrectly - the EDF file header will indicate the file is continuous (EDF+C) but the file is actually discontinuous (EDF+D)
	* within data2bids the error associated with this will state "File is not valid EDF+ or BDF+"
	* within data2bids, click on Tools --> Overwrite EDF type. In the pop-up window choose the problem file, and select the EDF+D option. The issue with the file is that Natus has inccorectly called the file EDF+C so this tool will overwrite this information to EDF+D.
	* now open EDFbrowser and use the tool "Convert EDF+D to EDF+C" to convert the file to EDF+C.

## [03.11.2021] (2021-03-11)

### Features

* a Pause button has been added so you are able to pause a conversion and resume later (note: if using this feature, data2bids needs to remain open while paused)
* Dark/Light mode has been added and can be toggled in File --> Theme
* an About data2bids menu has been added in the top navigation bar (provides links to the Google Drive folder and documentation website)
* when data2bids is opened, it will now automatically check the Google Drive folder for software updates
	* this check will always occur unless you specify otherwise in File --> Settings --> General settings --> check for updates on startup
