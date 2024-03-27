Quick Start
===========

If you installed via pip then `spydcmtk` console script will be exposed in your Python environment.

Access via:

.. code-block:: bash

    spydcmtk -h

to see the command-line usage available to you.

If you would like to incorporate `spydcmtk` into your Python project, then import as:

.. code-block:: python

    import spydcmtk

    listOfStudies = spydcmtk.dcmTK.ListOfDicomStudies.setFromDirectory(MY_DICOM_DIRECTORY)
    # Example filtering
    dcmStudy = listOfStudies.getStudyByDate('20230429') # Dates in DICOM standard string format: YYYYMMDD
    dcmSeries = dcmStudy.getSeriesBySeriesNumber(1)
    # Example writing new DICOM files with anonymization
    dcmStudy.writeToOrganisedFileStructure(tmpDir, anonName='Not A Name')

Configuration
-------------

`spydcmtk` uses a `spydcmtk.conf` file for configuration.

By default `spydcmtk.conf` files are searched for in the following locations:

1. `source_code_directory/spydcmtk.conf` (file with default settings)
2. `$HOME/spydcmtk.conf`
3. `$HOME/.spydcmtk.conf`
4. `$HOME/.config/spydcmtk.conf`
5. Full file path defined at environment variable: ``SPYDCMTK_CONF``
6. Full path passed as command-line argument to `spydcmtk`

Files are read in the above order with each subsequent variable present overwriting any previously defined.
For information on files found and variables used, run:

.. code-block:: bash

    spydcmtk -INFO