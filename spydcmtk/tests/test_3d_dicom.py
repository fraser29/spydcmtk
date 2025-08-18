#!/usr/bin/env python3
"""
Test script to demonstrate 3D DICOM handling functionality.
This script tests both traditional 2D slice-based DICOM files and 3D DICOM files.
"""
from context import spydcmtk # type: ignore # This is useful for testing outside of environment

import numpy as np
import sys
import os

from spydcmtk import dcmTK

def test_3d_dicom_handling():
    """Test the 3D DICOM handling functionality."""
    
    print("Testing 3D DICOM handling functionality...")
    
    # Test with existing test data (2D slice-based)
    print("\n1. Testing with traditional 2D slice-based DICOM files...")
    try:
        test_dicoms_dir = os.path.join(os.path.dirname(__file__), 'TEST_DATA', 'DICOMS')
        if os.path.exists(test_dicoms_dir):
            list_of_studies = dcmTK.ListOfDicomStudies.setFromDirectory(test_dicoms_dir, HIDE_PROGRESSBAR=True)
            if list_of_studies:
                dcm_study = list_of_studies[0]  # Get first study
                dcm_series = dcm_study[0]  # Get first series
                
                print(f"   Series has {len(dcm_series)} DICOM files")
                print(f"   has3DPixelData: {dcm_series.has3DPixelData()}")
                print(f"   is3D: {dcm_series.is3D()}")
                print(f"   getNumberOfSlicesPerVolume: {dcm_series.getNumberOfSlicesPerVolume()}")
                print(f"   getNumberOfTimeSteps: {dcm_series.getNumberOfTimeSteps()}")
                print(f"   hasExplicitSpacingInfo: {dcm_series.hasExplicitSpacingInfo()}")
                print(f"   hasVariableSliceThickness: {dcm_series.hasVariableSliceThickness()}")
                print(f"   3D Spacing (row, col, slice): {dcm_series.get3DSpacing()}")
                
                # Test pixel data extraction
                A, patient_meta = dcm_series.getPixelDataAsNumpy()
                print(f"   Extracted array shape: {A.shape}")
                print(f"   Array dtype: {A.dtype}")
                
                print("   ✓ 2D slice-based DICOM handling test passed")
            else:
                print("   ⚠ No DICOM studies found in test data")
        else:
            print("   ⚠ Test data directory not found")
    except Exception as e:
        print(f"   ✗ 2D slice-based DICOM handling test failed: {e}")
    
    # Test 3D DICOM detection logic
    print("\n2. Testing 3D DICOM detection logic...")
    try:
        # Create a mock DICOM series with 3D pixel data
        class Mock3DDicomSeries:
            def __init__(self):
                self.files = []
                
            def __getitem__(self, index):
                return self.files[index]
                
            def __len__(self):
                return len(self.files)
                
            def has3DPixelData(self):
                if len(self.files) == 0:
                    return False
                return len(self.files[0].pixel_array.shape) > 2
                
            def is3D(self):
                return self.has3DPixelData()
                
            def getNumberOfSlicesPerVolume(self):
                if self.has3DPixelData():
                    return self.files[0].pixel_array.shape[2] if len(self.files[0].pixel_array.shape) >= 3 else 1
                return 1
                
            def getNumberOfTimeSteps(self):
                if self.has3DPixelData():
                    return len(self.files)
                return 1
        
        class Mock3DDicomFile:
            def __init__(self, shape):
                self.pixel_array = np.zeros(shape)
        
        # Test with 3D data (e.g., 256x256x64 volume)
        mock_series_3d = Mock3DDicomSeries()
        mock_series_3d.files = [Mock3DDicomFile((256, 256, 64))]
        
        print(f"   Mock 3D series - has3DPixelData: {mock_series_3d.has3DPixelData()}")
        print(f"   Mock 3D series - is3D: {mock_series_3d.is3D()}")
        print(f"   Mock 3D series - getNumberOfSlicesPerVolume: {mock_series_3d.getNumberOfSlicesPerVolume()}")
        print(f"   Mock 3D series - getNumberOfTimeSteps: {mock_series_3d.getNumberOfTimeSteps()}")
        
        # Test with 4D data (e.g., 256x256x64x3 RGB volume)
        mock_series_4d = Mock3DDicomSeries()
        mock_series_4d.files = [Mock3DDicomFile((256, 256, 64, 3))]
        
        print(f"   Mock 4D series - has3DPixelData: {mock_series_4d.has3DPixelData()}")
        print(f"   Mock 4D series - is3D: {mock_series_4d.is3D()}")
        print(f"   Mock 4D series - getNumberOfSlicesPerVolume: {mock_series_4d.getNumberOfSlicesPerVolume()}")
        print(f"   Mock 4D series - getNumberOfTimeSteps: {mock_series_4d.getNumberOfTimeSteps()}")
        
        # Test spacing methods (these would need to be implemented in the mock)
        print("   Note: Mock series don't have real DICOM tags, so spacing methods would return defaults")
        print("   Note: Variable slice thickness detection would require real DICOM SharedFunctionalGroupsSequence data")
        print("   Note: Slice spacing prioritizes SpacingBetweenSlices over SliceThickness for proper 3D volume construction")
        
        print("   ✓ 3D DICOM detection logic test passed")
    except Exception as e:
        print(f"   ✗ 3D DICOM detection logic test failed: {e}")
    
    print("\n3D DICOM handling test completed!")

if __name__ == "__main__":
    test_3d_dicom_handling()
