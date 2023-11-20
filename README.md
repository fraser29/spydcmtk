# *spydcmtk*

*Simple PYthon DiCoM Tool Kit*

*spydcmtk* is a pure Python package built on top of [*pydicom*](https://github.com/pydicom/pydicom).

This package extends pydicom with a class structure based upon the Patient-Study-Series-Image heirachy. In addition, it provides a number of builtin routines for common actions when working with dicom files, such as human readable renaming, anonymisation, searching and summarising. 

## Version

Current is VERSION 1.0.1 Release. 

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
dcmStudy = listOfStudies.getStudyByDate('20140409') # Dates in dicom standard string format: YYYYMMDD
dcmSeries = dcmStudy.getSeriesBySeriesNumber(1)
dcmStudy.writeToOrganisedFileStructure(tmpDir, anonName='Not A Name')

```



## Documentation

Clear documentation of basic features can be seen by running the *"spycmtk -h"* command as referenced above. 

Some format conversions are provided by this package to permit further use of dicom data. 


### Dicom to VTK

A dicom to vtk format conversion is provided. See VTK format documentation [*here*](https://examples.vtk.org/site/VTKFileFormats/). 

Format conversions are: 

- dicom to image data (vti format). Suitable for 3D image volumes. This format is axis aligned (there is no embedded transformation). But "Field Data" embedded in the file are included as "ImageOrientationPatient" which, along with the Image Origin and Image Spacing methods can be used to construct a transformation matrix allowing conversion form image to real world coordinate space. 

- *WORK IN PROGRESS*: dicom to structured dataset (vts format). 

- *WORK IN PROGRESS*: dicom to planar dataset (vtp format). 

### Dicom to Nifti

Relies on [*dcm2nii*](https://people.cas.sc.edu/rorden/mricron/dcm2nii.html), which must be installed and in path.

### Dicom to HTML

Will build a standalone .html file to display dicom series in [*ParaView Glance*](https://www.kitware.com/exporting-paraview-scenes-to-paraview-glance/) renderer. 
