# *spydcmtk*

*Simple PYthon DiCoM Tool Kit*

Dicom organisational, querying and conversion toolkit

*spydcmtk* is a pure Python package built on top of [*pydicom*](https://github.com/pydicom/pydicom).

This package extends pydicom with a class structure based upon the Patient-Study-Series-Image hierarchy. In addition, it provides a number of built in routines for common actions when working with dicom files, such as human readable renaming, anonymisation, searching and summarising. 


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
to see the commandline usage available to you.


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

Files are read in the above order with each subsequent variable present over writing any previously defined. 
For information on files found and variables used run:

`spydcmtk -INFO` 


## Documentation

Clear documentation of basic features can be seen by running the *"spydcmtk -h"* command as referenced above. 
For detailed documentation please see [spydcmtk-documentation](https://fraser29.github.io/spydcmtk/)

### Converting PDF or images to DICOM

**PDF** (requires [DCMTK](https://dcmtk.org/) `pdf2dcm` on `PATH`):

```bash
spydcmtk -pdf2dcm /path/to/report.pdf -i /path/to/reference_dicom -o /path/to/output_dir
```

```python
import spydcmtk
spydcmtk.dcmTK.pdf2dcm("report.pdf", "reference.dcm", "output_dir")
```

**Image stack** (one file or a folder of `.jpg`/`.png`/`.tif` slices):

```bash
spydcmtk -image2dcm /path/to/images/ -i /path/to/reference_dicom -o /path/to/output_dir
```

```python
import spydcmtk
spydcmtk.dcmTK.writeImageStackToDicom(sorted_image_paths, patientMeta, "reference.dcm", "output_dir")
```

See the [DICOM conversion](https://fraser29.github.io/spydcmtk/usage/dicom-conversion.html) docs for `PatientMeta`, optional tag overrides, and CLI details.

