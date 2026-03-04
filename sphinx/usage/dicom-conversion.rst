DICOM CONVERSION
================

Some format conversions are provided by this package:

DICOM to Numpy Array
^^^^^^^^^^^^^^^^^^^^

DicomSeries class has a method getPixelDataAsNumpy() that returns a tuple of a numpy array and a PatientMeta object.

The returned numpy array is of shape [nColumns, nRows, nSlices, nTime], OR [nColumns, nRows, nSlices, nTime, nChannels].
The PatientMeta object contains metadata about the orientation of the image. 

Example:

.. code-block:: python
    import spydcmtk
    dcmSeries = spydcmtk.dcmTK.DicomSeries.setFromDirectory(dicom_directory, HIDE_PROGRESSBAR=True)
    A, patMeta = dcmSeries.getPixelDataAsNumpy()
    print(A.shape)
    plt.imshow(A[:,:,0,0])
    plt.show()


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

- A) DICOM to image data (vti format). Suitable for 3D image volumes. This format is axis-aligned (this is data in 3D space with no embedded transformation). But "Field Data" embedded in the file are included as "ImageOrientationPatient" which, along with the Image Origin and Image Spacing methods can be used to construct a transformation matrix allowing conversion from image to real-world coordinate space. *See "WORKING WITH PATIENT COORDINATES" section below for more details.* The slice axis will correspond to the ImageData z-axis (x-y is inplane - irrespective of the acquisition orientation).
- B) DICOM to structured dataset (vts format). This will result in a dataset that is in the true real world (patient) coordinate space. NOTE: Due to VTK format such a file has a much larger (disc) size than a VTI file.  
- C) DICOM to image data format but in the true image coordinates. This will be an axis-aligned image but covering the true image coordinates - image dimensions will be different from the original. This is basically a resample of the VTS output to a VTI dataset. If your DICOM data is axis-aligned then this is a good option.
- D) DICOM to image data format (as (A)) but with embedded direction axes. **NOTE:** This has variable support in visualisation and markup software. 

- VTI image data to DICOM is supported. But exact coordinate information may be lost due to the nature of the conversion. Due to the conversion steps the output DICOMS may be sliced along a different axis to the original.

WORKING WITH PATIENT COORDINATES
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For general DICOM manipulation and analysis the class PatientMeta found in dcmVTKTK.py is useful. It provides a helpful interface to move between image and patient coordinates. It also provides a number of properties that can be used to get the basic dicom meta data. This should be used if working with option (A) above. 

Example:

.. code-block:: python
    import spydcmtk
    dcmSeries = spydcmtk.dcmTK.DicomSeries.setFromDirectory(dicom_directory, HIDE_PROGRESSBAR=True)
    A, patMeta = dcmSeries.getPixelDataAsNumpy()
    xyz = patMeta.imageToPatientCoordinates(np.array([I, J, K]))
    print(xyz)