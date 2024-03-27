DICOM CONVERSION
================

Some format conversions are provided by this package:

DICOM to Nifti
^^^^^^^^^^^^^^

Relies on `dcm2niix <https://github.com/rordenlab/dcm2niix>`_, which must be installed and in path (or set in config file).

DICOM to HTML
^^^^^^^^^^^^^

Will build a standalone .html file to display DICOM series in `ParaView Glance <https://www.kitware.com/exporting-paraview-scenes-to-paraview-glance/>`_ renderer.

DICOM to VTK
^^^^^^^^^^^^

A DICOM to VTK format conversion is provided. See VTK format documentation `here <https://examples.vtk.org/site/VTKFileFormats/>`_.

Format conversions are:

- DICOM to image data (vti format). Suitable for 3D image volumes. This format is axis-aligned (there is no embedded transformation). But "Field Data" embedded in the file are included as "ImageOrientationPatient" which, along with the Image Origin and Image Spacing methods can be used to construct a transformation matrix allowing conversion from image to real-world coordinate space.
- Work In Progress: DICOM to structured dataset (vts format).
- Work In Progress: DICOM to planar dataset (vtp format).
