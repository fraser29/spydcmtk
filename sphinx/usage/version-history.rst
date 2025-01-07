Version
=======


Current is VERSION 1.2.2 Release. 

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