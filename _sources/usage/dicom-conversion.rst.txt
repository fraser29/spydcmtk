DICOM CONVERSION
================

Some format conversions are provided by this package.

Conversions **to** DICOM (from PDF or image stacks) are described in the **PDF to DICOM** and **Image stack to DICOM** sections below.

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
.. code-block:: bash
    spydcmtk -i input_directory -o output_directory -vti
- B) DICOM to structured dataset (vts format). This will result in a dataset that is in the true real world (patient) coordinate space. NOTE: Due to VTK format such a file has a much larger (disc) size than a VTI file.  
.. code-block:: bash
    spydcmtk -i input_directory -o output_directory -vts
- C) DICOM to image data format but in the true image coordinates. This will be an axis-aligned image but covering the true image coordinates - image dimensions will be different from the original. This is basically a resample of the VTS output to a VTI dataset. If your DICOM data is axis-aligned then this is a good option.
.. code-block:: bash
    spydcmtk -i input_directory -o output_directory -vti -TRUE_VTI_ORIENTATION
- D) DICOM to image data format (as (A)) but with embedded direction axes. **NOTE:** This has variable support in visualisation and markup software. In _BETA_ mode.
.. code-block:: bash
    spydcmtk -i input_directory -o output_directory -vti -DIRECTION_VECTORS


- VTI image data to DICOM is supported. But exact coordinate information may be lost due to the nature of the conversion. Due to the conversion steps the output DICOMS may be sliced along a different axis to the original.

PDF to DICOM
^^^^^^^^^^^^

Encapsulate a PDF as a DICOM Encapsulated PDF object, using a reference DICOM file to supply patient and study metadata.

**Requirements:** `DCMTK <https://dcmtk.org/>`_ must be installed and the ``pdf2dcm`` executable must be on your ``PATH``.

Command line
""""""""""""

Pass the PDF path with ``-pdf2dcm``, a template DICOM (file or directory containing at least one DICOM) with ``-i``, and the output directory with ``-o``:

.. code-block:: bash

    spydcmtk -pdf2dcm /path/to/report.pdf -i /path/to/reference_dicom_or_dir -o /path/to/output_dir

If ``-i`` points to a directory, the first DICOM found in that directory is used as the template. The output file is named ``{pdf_basename}.dcm`` inside the output directory.

Python script
"""""""""""""

.. code-block:: python

    import spydcmtk

    spydcmtk.dcmTK.pdf2dcm(
        "/path/to/report.pdf",
        dcmTemplateFile_or_ds="/path/to/reference.dcm",  # or a pydicom Dataset
        outputDir="/path/to/output_dir",
        tagUpdateDict={
            # Optional: DCMTK-style tags (group,element as hex string)
            "SeriesDescription": ["0008,103e", "LO", "My PDF report"],
        },
    )

Alternatively, use the thin wrapper in ``spydcmtk.spydcm`` (same arguments as the CLI):

.. code-block:: python

    import spydcmtk

    spydcmtk.spydcm.pdf2dcm(
        "/path/to/report.pdf",
        "/path/to/reference_dicom_or_dir",
        "/path/to/output_dir",
    )

Optional ``tagUpdateDict`` entries use the form ``{name: ["gggg,eeee", "VR", value]}``. Default tags (study date, accession number, series number 999, series description derived from the PDF filename) are applied automatically and can be overridden.

Image stack to DICOM
^^^^^^^^^^^^^^^^^^^^

Convert one or more raster images (JPEG, PNG, or TIFF) into a multi-slice DICOM series. Each image becomes one slice. Pixel bit depth and colour (greyscale vs RGB) are preserved from the source images where possible.

The Python API is ``writeImageStackToDicom``; the CLI flag is ``-image2dcm``.

Command line
""""""""""""

Single image file:

.. code-block:: bash

    spydcmtk -image2dcm /path/to/slice.png -i /path/to/reference.dcm -o /path/to/output_dir

Directory of images (``.jpg``, ``.png``, ``.tif``, ``.tiff``; files are sorted by name and stacked along the slice axis):

.. code-block:: bash

    spydcmtk -image2dcm /path/to/image_folder/ -i /path/to/reference_dicom_or_dir -o /path/to/output_dir

As with PDF conversion, ``-i`` may be a single DICOM file or a directory (first DICOM found is used as the template).

Python script
"""""""""""""

Provide a **sorted** list of image paths, a ``PatientMeta`` object describing geometry (origin, spacing, orientation), and a template DICOM:

.. code-block:: python

    import os
    import spydcmtk

    image_dir = "/path/to/slices"
    file_list = sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".tif", ".tiff"))
    )

    patient_meta = spydcmtk.dcmVTKTK.PatientMeta()
    patient_meta.initFromDictionary({
        "Origin": [0.0, 0.0, 0.0],
        "Spacing": [0.001, 0.001, 0.02],  # metres: in-plane x, in-plane y, between slices
        "ImageOrientationPatient": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    })

    spydcmtk.dcmTK.writeImageStackToDicom(
        file_list,
        patientMeta=patient_meta,
        dcmTemplateFile_or_ds="/path/to/reference.dcm",
        outputDir="/path/to/output_dir",
        tagUpdateDict=None,           # optional pydicom-style tag overrides
        CONVERT_TO_GREYSCALE=True,    # set False to keep RGB (CLI uses False)
    )

CLI wrapper (directory or single file, default geometry when ``patientMeta`` is not set):

.. code-block:: python

    import spydcmtk

    spydcmtk.spydcm.image2dcm(
        "/path/to/image_or_folder",
        "/path/to/reference_dicom_or_dir",
        "/path/to/output_dir",
    )

For full control over slice spacing and patient orientation, use ``writeImageStackToDicom`` with ``PatientMeta`` rather than the CLI wrapper. See `WORKING WITH PATIENT COORDINATES`_ for how ``PatientMeta`` relates image and patient space.

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