# *spydcmtk*

*Simple PYthon DiCoM Tool Kit*

Dicom organisational, querying and conversion toolkit

*spydcmtk* is a pure Python package built on top of [*pydicom*](https://github.com/pydicom/pydicom).

This package extends pydicom with a class structure based upon the Patient-Study-Series-Image hierarchy. In addition, it provides a number of built in routines for common actions when working with dicom files, such as human readable renaming, anonymisation, searching and summarising. 

## Version

Current is VERSION 1.1.3 Release. 

1.1.3: Additional configuration moved to config file. DCM2VTI active. 
1.1.1: Add option to keep private tags when running anonymisation. Dcm2nii path configurable from config file. 
1.1.0: Some bug fixes and restrict the use of dicom to vti (WIP)
1.0.0: Initial Release

## Installation

Using [pip](https://pip.pypa.io/en/stable/):
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
dcmStudy = listOfStudies.getStudyByDate('20230429') # Dates in dicom standard string format: YYYYMMDD
dcmSeries = dcmStudy.getSeriesBySeriesNumber(1)
dcmStudy.writeToOrganisedFileStructure(tmpDir, anonName='Not A Name')

```


# Configuration

miresearch uses a spydcmtk.conf file for configuration. 

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

For detailed documentation please see [wiki](https://github.com/fraser29/spydcmtk/wiki)

Some format conversions are provided by this package:

### Dicom to Nifti

Relies on [*dcm2niix*](https://github.com/rordenlab/dcm2niix), which must be installed and in path (or set in config file).

### Dicom to HTML

Will build a standalone .html file to display dicom series in [*ParaView Glance*](https://www.kitware.com/exporting-paraview-scenes-to-paraview-glance/) renderer. 


### Dicom to VTK

A dicom to vtk format conversion is provided. See VTK format documentation [*here*](https://examples.vtk.org/site/VTKFileFormats/). 

Format conversions are: 

- dicom to image data (vti format). Suitable for 3D image volumes. This format is axis aligned (there is no embedded transformation). But "Field Data" embedded in the file are included as "ImageOrientationPatient" which, along with the Image Origin and Image Spacing methods can be used to construct a transformation matrix allowing conversion form image to real world coordinate space. 

- *WORK IN PROGRESS*: dicom to structured dataset (vts format). 

- *WORK IN PROGRESS*: dicom to planar dataset (vtp format).

