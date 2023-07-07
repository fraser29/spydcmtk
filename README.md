# *spydcmtk*

*Simple PYthon DiCoM Tool Kit*

*spydcmtk* is a pure Python package built on top of [*pydicom*](https://github.com/pydicom/pydicom).

This package extends pydicom with a class structure based upon the Patient-Study-Series-Image heirachy. In addition, it provides a number of builtin routines for common actions when working with dicom files, especially in a research based environment. 

## Installation

Using [pip](https://pip.pypa.io/en/stable/):
```
# TODO
# pip install spydcmtk
```

## Quick start

If you installed via 'setup.py' then spydcmtk console script will be exposed in your python environment. 

Access via:
```bash
spydcmtk -h
```
to see the commandline useage available to you.

If you would like to incorporate spydcmtk into your python project, then import as:
```python
import spydcmtk

# TODO : Example to read and work with a dicom files via:

```



## Documentation



Some format conversions are provided by this package to permit further use of dicom data. 


### Dicom to VTK

A dicom to vtk format conversion is provided. See VTK format documentation [*here*](https://examples.vtk.org/site/VTKFileFormats/). 

Format conversions are: 

- dicom to image data (vti format). Suitable for 3D image volumes. This format is axis aligned (there is no embedded transformation). But "Field Data" embedded in the file are included as "ImageOrientationPatient" which, along with the Image Origin and Image Spacing methods can be used to construct a transformation matrix allowing conversion form image to real world coordinate space. 

- dicom to structured dataset (vts format). 

- dicom to planar dataset (vtp format). 