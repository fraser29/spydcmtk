Tutorial
===========

For most users, it is sufficient to use `spydcmtk` via the command-line interface.

Access via:

.. code-block:: bash

    spydcmtk -h

to see the command-line usage available to you.

Inspect and set tags (CLI)
--------------------------

Fast inspection of organised DICOM trees. Reads **one DICOM per series directory**, then prints subject, study, and **series** info for **every study** under the input path:

.. code-block:: bash

    spydcmtk -i /path/to/dicoms -inspectQuick

Set a DICOM tag on all instances found under the input and write an organised copy to ``-o`` (required; must differ from ``-i``). If multiple studies are present, each is updated:

.. code-block:: bash

    spydcmtk -i /path/to/dicoms -o /path/to/output -setTag SeriesDescription "My new description"
    spydcmtk -i /path/to/dicoms -o /path/to/output -setTag StudyDescription "Updated study"

The same operation from a script:

.. code-block:: python

    import spydcmtk

    # Convenience wrapper (reads all studies under path, writes organised output)
    spydcmtk.spydcm.setTag(
        "/path/to/dicoms",
        "SeriesDescription",
        "My new description",
        "/path/to/output",
    )

    # Or on an already-loaded study / series
    study = spydcmtk.dcmTK.DicomStudy.setFromDirectory("/path/to/study")
    study.setTags_all("SeriesDescription", "My new description")
    study.writeToOrganisedFileStructure("/path/to/output")

Interactive mode
----------------

Launch a menu for a study (optionally pass ``-o`` for a default output directory):

.. code-block:: bash

    spydcmtk -i /path/to/dicoms -o /path/to/output -I

Menu options include data summary, Build VTS, Build FDQ, Build overview image, **Build NII**, and **Set tag**. For Build NII and Set tag you can target a series number or ``all``.

If you would like to incorporate `spydcmtk` into your Python project, then some tutorial snippets showing primary functionality are provided below.

.. code-block:: python

    import spydcmtk

    listOfStudies = spydcmtk.dcmTK.ListOfDicomStudies.setFromDirectory(MY_DICOM_DIRECTORY)
    # Example filtering
    dcmStudy = listOfStudies.getStudyByDate('20230429') # Dates in DICOM standard string format: YYYYMMDD
    dcmSeries = dcmStudy.getSeriesBySeriesNumber(1)
    # Example writing new DICOM files with anonymization
    dcmStudy.writeToOrganisedFileStructure(tmpDir, anonName='Not A Name')



Work with pixel data and metadata
----------------------------------

.. code-block:: python

    listOfStudies = spydcmtk.dcmTK.ListOfDicomStudies.setFromDirectory(MY_DICOM_DIRECTORY)
    dcmStudy = listOfStudies[0]
    dcmSeries = dcmStudy.getSeriesBySeriesNumber(4)
    A, patientMeta = dcmSeries.getPixelDataAsNumpy()
    print(A.shape)
    print(patientMeta)
    # OUTPUT (e.g.):            
    # 'ImageOrientationPatient': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    # 'PixelSpacing': [1.0, 1.0],
    # 'SliceThickness': 1.0,
    # 'ImagePositionPatient': [0.0, 0.0, 0.0],
    # 'SliceLocation0': 0.0



Convert image(s) to DICOM
--------------------------

.. code-block:: python

    pixArray # A numpy array of shape (nC, nR, nS, nChannels)
    # Expand dimensions to include a single channel dimension if necessary
    pixArray = np.expand_dims(pixArray, 3)

    # Define image spacing / orientation / thickness
    patMatrix = {'PixelSpacing': [0.02, 0.02], 
                    'ImagePositionPatient': [0.0, 0.1, 0.3], 
                    'ImageOrientationPatient': [0.0,0.0,1.0,0.0,1.0,0.0], 
                    'SliceThickness': 0.04,
                    'SpacingBetweenSlices': 0.04}
    patMeta = dcmTK.dcmVTKTK.PatientMeta()
    # Initialise patient meta from dictionary - optional - defaults will be used if not provided
    patMeta.initFromDictionary(patMatrix)
    
    # Define tag values to update: Note Name:Value or Name:(Code, VR, Value) is supported
    tagUpdateDict = {'SeriesNumber': 88, 
                        'StudyDescription': ([0x0008,0x1030], 'LO', "TestDataB"), 
                        'SeriesDescription': ([0x0008,0x103e], 'LO', "SeriesLaugh"), 
                        'StudyID': ([0x0020,0x0010], 'SH', '1088')}
    
    # DCM Template - Optional - will use defaults if not provided. May be filepath or pydicom dataset.
    dcmTemplate = None
    # OR
    dcmTemplate = 'path/to/dcm/template.dcm'
    # OR
    dcmTemplate = dcm_series_object[0]

    # Write to DICOM
    dcmTK.writeNumpyArrayToDicom(pixelArrray=pixArray, 
                                dcmTemplate_or_ds=dcmTemplate, 
                                patientMeta=patMeta, 
                                outputDir=tmpDir, 
                                tagUpdateDict=tagUpdateDict)


Anonymise multiple data for blinded reading - then reidentify
-------------------------------------------------------------

Assume "WORKING_DIR" is a directory containing multiple studies, each with multiple series. 
Such a structure is typical output from a commanline action:

.. code-block:: bash

    spydcmtk -i DICOM_DIR -o WORKING_DIR


.. code-block:: python

    # Read each study individually (save memory)
    matchingList = []
    for k1, iDir in enumerate(sorted(os.listdir(WORKING_DIR))):
        fullDir = os.path.join(WORKING_DIR, iDir)
        print(f"Working on: {fullDir}")
        dmcStudy = spydcmtk.dcmTK.DicomStudy.setFromDirectory(fullDir, HIDE_PROGRESSBAR=True)
        filteredList = dmcStudy.filterByTag("SeriesDescription", "T1-Head") # Filter by series description
        for iSeries in filteredList:
            originalName = iSeries.getTag('PatientName')
            thisSeriesUID = iSeries.getTag('SeriesInstanceUID')
            newName = f"anon_for_readerA_{k1}"
            newPatID = k1
            iSeries.anonymise(newName, newPatID)
            newStudyUID = spydcmtk.dcmTK.generate_uid()
            iSeries.resetUIDs(newStudyUID)
            newSeriesUID = iSeries.getTag('SeriesInstanceUID')
            newStudyDir = os.path.join(WORKING_ANON, f"anon_for_readerA_{k1}")
            os.makedirs(newStudyDir, exist_ok=True)
            dirOut = iSeries.writeToOrganisedFileStructure(newStudyDir)
            matchingList.append([originalName, thisSeriesUID, newName, newPatID, newSeriesUID])
    # now, e.g., write matchinglist to csv file for later reidentification. 

