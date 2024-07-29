# *spydcmtk*

*Simple PYthon DiCoM Tool Kit*

Dicom organisational, querying and conversion toolkit

*spydcmtk* is a pure Python package built on top of [*pydicom*](https://github.com/pydicom/pydicom).

This package extends pydicom with a class structure based upon the Patient-Study-Series-Image hierarchy. In addition, it provides a number of built in routines for common actions when working with dicom files, such as human readable renaming, anonymisation, searching and summarising. 

## Version

Current is VERSION 1.1.8 Release. 

- 1.1.8: Added improved functionality for dicom to: VTK image data; and VTK structured grid data conversion
- 1.1.7: Add basic DCM-SEG read/write/conversion functionality. Rewrote dcm2vtk routines for improved consistency in some edge cases. 
- 1.1.5: Add option to retrieve tag value from commandline. Small bug fix on safe naming. 
- 1.1.4: Additional configuration moved to config file. DCM2VTI active. 
- 1.1.1: Add option to keep private tags when running anonymisation. Dcm2nii path configurable from config file. 
- 1.1.0: Some bug fixes and restrict the use of dicom to vti (WIP)
- 1.0.0: Initial Release

## Installation

Using [pip](https://pypi.org/project/spydcmtk/):
```
pip install spydcmtk
```

## Quick start

If you installed via pip then *spydcmtk* console script will be exposed in your python environment. 

Access via:
```bash
spydcmtk -h
```
to see the commandline useage available to you.


If you would like to incorporate spydcmtk into your python project, then import as:
```python
import spydcmtk

listOfStudies = spydcmtk.dcmTK.ListOfDicomStudies.setFromDirectory(MY_DICOM_DIRECTORY)
# Example filtering
dcmStudy = listOfStudies.getStudyByDate('20230429') # Dates in dicom standard string format: YYYYMMDD
dcmSeries = dcmStudy.getSeriesBySeriesNumber(1)
# Example writing new dicom files with anonymisation
dcmStudy.writeToOrganisedFileStructure(tmpDir, anonName='Not A Name')

```


# Configuration

spydcmtk uses a spydcmtk.conf file for configuration. 

By default spydcmtk.conf files are search for in the following locations: 

1. source_code_directory/spydcmtk.conf (file with default settings)
2. $HOME/spydcmtk.conf
3. $HOME/.spydcmtk.conf
4. $HOME/.config/spydcmtk.conf
5. Full file path defined at environment variable: "SPYDCMTK_CONF"
6. Full path passed as commandline argument to `spydcmtk`

Files are read in the above order with each subsequent variable present overwritting any previously defined. 
For information on files found and variables used run:

`spydcmtk -INFO` 


## Documentation

Clear documentation of basic features can be seen by running the *"spycmtk -h"* command as referenced above. 
For detailed documentation please see [spydcmtk-documnetation](https://fraser29.github.io/spydcmtk/)

## Works in progress

Basic handling of DicomSegmentation images is added :

- [x] writing a label map to dicom segmentation
- [x] basic reading a dicom segmentation and writing to vti/vts
- [ ] handling multi label dicom segmentations
- [ ] improved, explicit handling of data orientation for
  - [ ] multi-volume dicom images
  - [ ] multi-label dicom seg images 