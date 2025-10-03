# Version History

## [1.2.15] - 2025-10-01
### Feature
Force FieldData - ImageOrientationPatient [1,0,0,0,1,0] - for VTI TrueOrientation

## [1.2.14] - 2025-09-30
### Fixed
- Fix bug DICOM to VTI orientation - swap rows and columns to ensure coordinates correctly converted back to real world units by PatientMatrix. 

## [1.2.13] - 2025-09-10
### Fixed
- Fix bug in writeNumpyArrayToDicom - correctly handle pixelArray with negative values.

## [1.2.12] - 2025-08-25
### Fixed
- Fix bug in vti2dcm command - correctly handle file read errors.

## [1.2.11] - 2025-08-18
### Added
- 3D DICOM file handling.

## [1.2.10] - 2025-04-28
### Added
- Add warning for EXTRA_TAGS deprecation in method DicomStudy.getSeriesInfoDict.
- Added CHANGELOG.md

### Fixed
- Combine EXTRA_TAGS and extraTags parameters in method DicomStudy.getSeriesInfoDict.


## [1.2.9] - 2025-04-25
### Fixed
- Fix bug in some tag retrieval methods via code. 

## PREVIOUS VERSIONS

- 1.2.5: Fix 4DFlow edge case bug. 
- 1.2.4: Small update to avoid edge case when concurrent dcm2VT* conversions within same directory. 4DFlow velocity output set to m/s by default. 
- 1.2.3: Fix VTI to DICOM and add to script. Update tests. Fix jpg to DICOM. nii2dcm also handled but nii orientation is not adjusted from RAS to LPS. 
- 1.2.2: Fix DICOM to VTK conversion bug. Add 4DFlow MRI to VTK conversion capability. 
- 1.2.1: Add filter by tag name and value. Add build image overview option. Update to use pydicom >=3.0.1
- 1.2.0: Improved stability of VTK conversion. Bug fixes. Add basic interactive functionality. Add functionality to construct 4D-flow datasets. Add reliance on external library ngawari for basic IO operations, format conversion and vtk filter actions. 
- 1.1.9: Permit user naming of series directory when writing at series level. Assistance for modifying tag values. 
- 1.1.8: Added improved functionality for dicom to: VTK image data; and VTK structured grid data conversion
- 1.1.7: Add basic DCM-SEG read/write/conversion functionality. Rewrote dcm2vtk routines for improved consistency in some edge cases. 
- 1.1.5: Add option to retrieve tag value from commandline. Small bug fix on safe naming. 
- 1.1.4: Additional configuration moved to config file. DCM2VTI active. 
- 1.1.1: Add option to keep private tags when running anonymisation. Dcm2nii path configurable from config file. 
- 1.1.0: Some bug fixes and restrict the use of dicom to vti (WIP)
- 1.0.0: Initial Release