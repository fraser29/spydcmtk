spydcmtk
========

Simple PYthon DiCoM Tool Kit

Dicom organisational, querying and conversion toolkit


Overview
--------

`spydcmtk` is a Python module designed for working with DICOM (Digital Imaging and Communications in Medicine) data. It builds upon `pydicom`, extending its functionality with an object-oriented approach based on the patient, study, series, image hierarchy commonly found in medical imaging. This allows for high-level functionality for reading, writing, and manipulating DICOM files.

Features
--------

- **Object-Oriented Scheme**: `spydcmtk` utilizes an object-oriented scheme based on the patient, study, series, image hierarchy of medical imaging.
  
- **Enhanced Functionality**: Extends `pydicom` to provide enhanced functionality for handling DICOM files, including reading, writing, and manipulation.

- **High-Level Abstractions**: Provides high-level abstractions for common DICOM operations, allowing for easier management of DICOM data.

- **Integration with pydicom**: Built on top of `pydicom`, ensuring compatibility and leveraging its existing features.

- **Support for Medical Imaging Workflows**: Designed to support common workflows in medical imaging, such as organizing, querying, and converting DICOM data.

Usage
-----

To use `spydcmtk`, first install it using `pip`:

.. code-block:: bash

    pip install spydcmtk

Once installed, you can import the module and utilize its functionality:

.. code-block:: python

    import spydcmtk

    # Example usage
    listOfStudies = spydcmtk.dcmTK.ListOfDicomStudies.setFromDirectory(MY_DICOM_DIRECTORY)
    # Example filtering
    dcmStudy = listOfStudies.getStudyByDate('20230429') # Dates in DICOM standard string format: YYYYMMDD
    dcmSeries = dcmStudy.getSeriesBySeriesNumber(1)
    # Example writing new DICOM files with anonymization
    dcmStudy.writeToOrganisedFileStructure(tmpDir, anonName='Not A Name')


Contributing
------------

If you'd like to contribute to `spydcmtk`, please check out the project repository on GitHub and consider submitting pull requests, reporting issues, or contributing code/documentation improvements.

GitHub Repository: https://github.com/fraser29/spydcmtk

License
-------

`spydcmtk` is distributed under the XYZ license. Please refer to the LICENSE file in the project repository for more information.



`spydcmtk` is a pure Python package built on top of `pydicom <https://github.com/pydicom/pydicom>`_.

This package extends pydicom with a class structure based upon the Patient-Study-Series-Image hierarchy. In addition, it provides a number of built-in routines for common actions when working with DICOM files, such as human-readable renaming, anonymization, searching, and summarizing.



Installation
------------

Using `pip`:

.. code-block:: bash

    pip install spydcmtk

