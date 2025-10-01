"""
Created on MArch 2023 (rewrite from old module - remove reliance on VTKDICOM)

@author: fraser

Dicom to VTK conversion toolkit

"""

import os
import numpy as np
from typing import Optional, Dict, List, Any
import pydicom as dicom
from pydicom.sr.codedict import codes
from pydicom.uid import generate_uid
from highdicom.seg.content import SegmentDescription
from highdicom.seg.enum import SegmentAlgorithmTypeValues, SegmentationTypeValues
from highdicom.content import AlgorithmIdentificationSequence
from highdicom.seg.sop import Segmentation

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import vtk
from vtk.util import numpy_support # type: ignore

import spydcmtk.dcmTools as dcmTools
from ngawari import fIO
from ngawari import ftk
from ngawari import vtkfilters
from multiprocessing import Pool


mm_to_m = 0.001
cm_to_m = 0.01
ms_to_s = 0.001
m_to_mm = 1000.0

# =========================================================================
## PATIENT MATRIX HELPER
# =========================================================================
class PatientMeta:
    """A class that manages spatial / geometric information for DICOM and VTK conversion
    
    _meta keys:
    'ImagePositionPatient', 'PixelSpacing', 'ImageOrientationPatient', 'SliceVector', 'SliceThickness', 'SliceLocation0', 'SpacingBetweenSlices', 'Dimensions'
    """

    def __init__(self):
        self.units = "SI"
        # Minimal defaults
        self._meta = {
            'ImageOrientationPatient': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            'PixelSpacing': [1.0, 1.0],
            'SliceThickness': 1.0,
            'ImagePositionPatient': [0.0, 0.0, 0.0],
            'SliceLocation0': 0.0
            }
        self._matrix = None

    def __str__(self):
        return str(self._meta)

    # Properties
    @property
    def PixelSpacing(self):
        return self._meta['PixelSpacing']
    
    @property
    def ImagePositionPatient(self):
        return self._meta['ImagePositionPatient']
    
    @property
    def ImageOrientationPatient(self):
        return self._meta['ImageOrientationPatient']
    
    @property
    def SliceVector(self):
        try:
            return self._meta['SliceVector']
        except KeyError:
            return np.cross(self._meta['ImageOrientationPatient'][:3], self._meta['ImageOrientationPatient'][3:6])
    
    @property
    def SpacingBetweenSlices(self):
        try:
            return self._meta['SpacingBetweenSlices']
        except KeyError:
            return self._meta['SliceThickness']
    
    @property
    def SliceThickness(self):
        try:
            return self._meta['SliceThickness']
        except KeyError:
            return self._meta['SpacingBetweenSlices']
    
    @property
    def SliceLocation0(self):
        try:
            return self._meta['SliceLocation0']
        except KeyError:
            return 0.0
    
    @property
    def Dimensions(self):
        return self._meta['Dimensions']

    @property
    def Origin(self):
        return self.ImagePositionPatient
    
    @property
    def Spacing(self):
        if 'SpacingBetweenSlices' in self._meta:
            return self._meta['PixelSpacing'][0], self._meta['PixelSpacing'][1], self._meta['SpacingBetweenSlices']
        else:
            return self._meta['PixelSpacing'][0], self._meta['PixelSpacing'][1], self._meta['SliceThickness']
    
    @property
    def PatientPosition(self):
        try:
            return self._meta['PatientPosition']
        except KeyError:
            return 'HFS'

    @property
    def Times(self):
        try:
            return self._meta['Times']
        except KeyError:
            return [0.0]

    # ------------------------------------------------------------------------------------------------------------------------------
    def initFromDictionary(self, metaDict):
        # Force required keys
        if 'PixelSpacing' not in metaDict:
            if 'Spacing' not in metaDict:
                metaDict['Spacing'] = [1.0, 1.0, 1.0]
        if 'ImagePositionPatient' not in metaDict:
            if 'Origin' not in metaDict:
                metaDict['Origin'] = [0.0, 0.0, 0.0]
        if 'ImageOrientationPatient' not in metaDict:
            metaDict['ImageOrientationPatient'] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        self._meta.update(metaDict)
        self.__metaVTI2DCMConversion()
        self._updateMatrix()

    def __metaVTI2DCMConversion(self):
        if 'Spacing' in self._meta:
            self._meta['PixelSpacing'] = [self._meta['Spacing'][0], self._meta['Spacing'][1]]
            if len(self._meta['Spacing']) > 2:
                self._meta['SliceThickness'] = self._meta['Spacing'][2]
        if 'SpacingBetweenSlices' not in self._meta:
            self._meta['SpacingBetweenSlices'] = self._meta['SliceThickness']
        if 'Origin' in self._meta:
            self._meta['ImagePositionPatient'] = [self._meta['Origin'][0], self._meta['Origin'][1], self._meta['Origin'][2]]

    def initFromDicomSeries(self, dicomSeries):
        # Check if we have 3D DICOM files (pixel data with more than 2 dimensions)
        firstPixelArray = dicomSeries[0].pixel_array
        is3DPixelData = len(firstPixelArray.shape) > 2
        
        if is3DPixelData:
            # Handle 3D DICOM files - each file contains a 3D volume
            I, J, K = firstPixelArray.shape[:3]  # First 3 dimensions are spatial
            N = len(dicomSeries)  # Number of time steps = number of files
            
            # Check if we have a 4th dimension (e.g., RGB channels)
            if len(firstPixelArray.shape) == 4:
                # Handle RGB/RGBA data
                channels = firstPixelArray.shape[3]
                A = np.zeros((I, J, K, N, channels), dtype=firstPixelArray.dtype)
                for n in range(N):
                    A[:, :, :, n, :] = dicomSeries[n].pixel_array
            else:
                # Standard 3D data
                A = np.zeros((I, J, K, N), dtype=firstPixelArray.dtype)
                for n in range(N):
                    A[:, :, :, n] = dicomSeries[n].pixel_array
        else:
            # Handle traditional 2D slice-based DICOM files
            I, J, K = int(dicomSeries.getTag('Rows')), int(dicomSeries.getTag('Columns')), int(dicomSeries.getNumberOfSlicesPerVolume())
            dicomSeries.sortBySlice_InstanceNumber()
            N = dicomSeries.getNumberOfTimeSteps()
            A = np.zeros((I, J, K, N))
            c0 = 0
            for k1 in range(K):
                for k2 in range(N):
                    iA = dicomSeries[c0].pixel_array
                    A[:, :, k1, k2] = iA
                    c0 += 1
        
        dt = dicomSeries.getTemporalResolution()
        if dt < 0.0000000001:
            dt = 1.0
        ipp = dicomSeries.getImagePositionPatient_np(0)
        sliceVec = dicomSeries.getSliceNormalVector()
        
        self._meta = {
                    'PixelSpacing': [dicomSeries.getDeltaRow()*mm_to_m, dicomSeries.getDeltaCol()*mm_to_m],
                    'SpacingBetweenSlices': dicomSeries.getDeltaSlice()*mm_to_m,
                    'SliceThickness': dicomSeries.getTag('SliceThickness', convertToType=float, ifNotFound=dicomSeries.getDeltaSlice())*mm_to_m,
                    'SliceLocation0': dicomSeries.getTag('SliceLocation', 0, ifNotFound=0.0, convertToType=float)*mm_to_m,
                    'ImagePositionPatient': [i*mm_to_m for i in ipp], 
                    'ImageOrientationPatient': dicomSeries.getImageOrientationPatient_np(0), 
                    'PatientPosition': dicomSeries.getTag("PatientPosition"), 
                    'Times': [dt*n*ms_to_s for n in range(N)], # ms to s
                    'Dimensions': A.shape,
                    'SliceVector': sliceVec,
                }
        self._updateMatrix()

    def initFromVTI(self, vtiObj, scaleFactor=1.0):
        dx,dy,dz = vtiObj.GetSpacing()
        oo = vtiObj.GetOrigin()
        # 1st option from meta, then field data then default
        try:
            iop = vtkfilters.getFieldData(vtiObj, 'ImageOrientationPatient')
        except AttributeError:
            iop = [1.0,0.0,0.0,0.0,1.0,0.0]
        try:
            sliceVec = vtkfilters.getFieldData(vtiObj, 'SliceVector')
        except AttributeError:
            sliceVec = [0.0,0.0,1.0]
        dims = [0,0,0]
        vtiObj.GetDimensions(dims)
        self._meta = {'PixelSpacing': [dy*scaleFactor, dx*scaleFactor],
                            'ImagePositionPatient': [i*scaleFactor for i in oo],
                            'ImageOrientationPatient': iop,
                            'SpacingBetweenSlices': dz*scaleFactor,
                            'SliceVector': sliceVec,
                            'Dimensions': dims,
                            'SliceThickness': dz*scaleFactor,
                            'SliceLocation0': 0.0}
        self._updateMatrix()

    def initFromDicomSeg(self, dicomSeg, scaleFactor=1.0): # TODO
        pass
        # sliceThick = dicomSeg.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness
        # pixSpace = dicomSeg.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        # ipp = [i.PlanePositionSequence[0].ImagePositionPatient for i in dicomSeg.PerFrameFunctionalGroupsSequence]
        # oo = np.array(ipp[0])
        # normalVector = np.array(ipp[-1]) - oo 
        # normalVector = normalVector / np.linalg.norm(normalVector)
        # oo = dicomSeg.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient
        # iop = dicomSeg.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient
        # seg_data = dicomSeg.pixel_array
        # seg_data = np.transpose(seg_data, axes=[2,1,0])
        # self._meta = {"ImagePositionPatient": [oo[0]*scaleFactor, oo[1]*scaleFactor, oo[2]*scaleFactor], 
        #         "PixelSpacing": [pixSpace[0]*scaleFactor, pixSpace[1]*scaleFactor],
        #         "SliceThickness": sliceThick*scaleFactor,
        #         "SpacingBetweenSlices": sliceThick*scaleFactor,
        #         "Dimensions": seg_data.shape,
        #         "ImageOrientationPatient": iop, 
        #         "SliceVector": normalVector   
        #         }
        # self._updateMatrix()

    def _updateMatrix(self):
        self._matrix = self.buildImage2PatientCoordinateMatrix()

    def buildImage2PatientCoordinateMatrix(self):
        dr, dc, dz = self.Spacing
        oo = self.ImagePositionPatient
        iop = np.array(self.ImageOrientationPatient)
        vZ = self.SliceVector
        matrix = np.array([
            [iop[0]*dc, iop[3]*dr, vZ[0]*dz, oo[0]], 
            [iop[1]*dc, iop[4]*dr, vZ[1]*dz, oo[1]], 
            [iop[2]*dc, iop[5]*dr, vZ[2]*dz, oo[2]], 
            [0, 0, 0, 1]
        ])
        return matrix

    def getMatrix(self):
        return self._matrix

    def getMetaForVTK(self):
        return {
            'Origin': self.Origin,
            'Spacing': self.Spacing,
            'ImageOrientationPatient': self.ImageOrientationPatient,
            'SliceVector': self.SliceVector,
            'Dimensions': self.Dimensions[:3]
        }

    def getMetaForDICOM(self):
        """
        Returns a dictionary with the meta data for DICOM
        Keys are: 
        'ImagePositionPatient', 'PixelSpacing', 'ImageOrientationPatient', 'SliceVector', 'SliceThickness', 'SliceLocation0', 'SpacingBetweenSlices'
        All in mm
        """
        return {
            'ImagePositionPatient': np.array([i*m_to_mm for i in self.ImagePositionPatient]),
            'PixelSpacing': np.array([i*m_to_mm for i in self.PixelSpacing]),
            'ImageOrientationPatient': self.ImageOrientationPatient,
            'SliceVector': self.SliceVector,
            'SliceThickness': self.SliceThickness*m_to_mm,
            'SliceLocation0': self.SliceLocation0*m_to_mm,
            'SpacingBetweenSlices': self.SpacingBetweenSlices*m_to_mm
        }   

    def imageToPatientCoordinates(self, imageCoords):
        if imageCoords.ndim == 1:
            homogeneous_coords = np.hstack((imageCoords, np.ones((1,))))
            return (self._matrix @ homogeneous_coords.T).T[:3]
        else:
            homogeneous_coords = np.hstack((imageCoords, np.ones((imageCoords.shape[0], 1))))
            return (self._matrix @ homogeneous_coords.T).T[:, :3]

    def patientToImageCoordinates(self, patientCoords):
        homogeneous_coords = np.hstack((patientCoords, np.ones((patientCoords.shape[0], 1))))
        return (np.linalg.inv(self._matrix) @ homogeneous_coords.T).T[:, :3]

    def getVtkMatrix(self):
        vtkMatrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtkMatrix.SetElement(i, j, self._matrix[i, j])
        return vtkMatrix

    def getVTKTransform(self):
        vtkMatrix = self.getVtkMatrix()
        vtkTransform = vtk.vtkTransform()
        vtkTransform.SetMatrix(vtkMatrix)
        return vtkTransform

    def transformVTKData(self, vtkData):
        transform = self.getVTKTransform()
        transformFilter = vtk.vtkTransformFilter()
        transformFilter.SetInputData(vtkData)
        transformFilter.SetTransform(transform)
        transformFilter.Update()
        return transformFilter.GetOutput()

    def getMeta(self):
        return self._meta

    def setMeta(self, key, value):
        self._meta[key] = value
        self._updateMatrix()

    def updateFromMeta(self, metaDict):
        self._meta.update(metaDict)
        self._updateMatrix()


# ===================================================================================================
# EXPOSED METHODS
# ===================================================================================================

def arrToVTI(arr: np.ndarray, 
            patientMeta: PatientMeta, 
            ds: Optional[dicom.Dataset] = None, 
            TRUE_ORIENTATION: bool = False,
            outputPath: Optional[str] = None) -> Dict[float, vtk.vtkImageData]:
    """Convert array (+meta) to VTI dict (keys=times, values=VTI volumes). 

    Args:
        arr (np.array): Array of pixel data, shape: nR,nC,nSlice,nTime
        patientMeta (PatientMatrix): PatientMatrix object containing meta to be added as Field data
        ds (pydicom dataset [optional]): pydicom dataset to use to add dicom tags as field data
        TRUE_ORIENTATION (bool [False]) : Boolean to force accurate spatial location of image data.
                                NOTE: this uses resampling from VTS data so output VTI will have different dimensions. 
        outputPath (str [None]): Save as go, then rename at end - RAM friendly
    
    Returns:
        vtiDict

    Raises:
        ValueError: If VTK import not available
    """
    dims = arr.shape
    if len(dims) == 3:
        arr = arr[..., np.newaxis]
        dims = arr.shape
    vtkDict = {}
    for k1 in range(dims[-1]):
        A3 = arr[:,:,:,k1]
        ###        
        try:
            thisTime = patientMeta.Times[k1]
        except KeyError:
            thisTime = k1
        #
        extra_tags = {"SliceVector": patientMeta.SliceVector,
                        "Time": thisTime}
        if TRUE_ORIENTATION:
            A3 = np.swapaxes(A3, 0, 1)
            vts_data = __arr3ToVTS(A3, patientMeta, ds, thisTime=thisTime)
            newImg = filterResampleToImage(vts_data, np.min(patientMeta.Spacing))
            vtkfilters.delArraysExcept(newImg, [], pointData=False)
            vtkfilters.delArraysExcept(newImg, ['PixelData'])
            extra_tags["ImageOrientationPatient"] = [1,0,0,0,1,0]
        else:
            A3 = np.swapaxes(A3, 0, 1)
            newImg = _arrToImagedata(A3, patientMeta)

        if ds is not None:
            addFieldDataFromDcmDataSet(newImg, ds, extra_tags=extra_tags)
        if outputPath is not None:
            newImg = fIO.writeVTKFile(newImg, os.path.join(outputPath, f"data_{generate_uid()}_{k1:05d}.vti"))
        vtkDict[thisTime] = newImg
    return vtkDict

def _arrToImagedata(A3: np.ndarray, patientMeta: PatientMeta) -> vtk.vtkImageData:
    newImg = _buildVTIImage(patientMeta)
    npArray = np.reshape(A3, np.prod(A3.shape), 'F').astype(np.int16)
    aArray = numpy_support.numpy_to_vtk(npArray, deep=1)
    aArray.SetName('PixelData')
    newImg.GetPointData().SetScalars(aArray)
    return newImg

def _buildVTIImage(patientMeta: PatientMeta=None) -> vtk.vtkImageData:
    if patientMeta is None:
        patientMeta = PatientMeta()
    vti_image = vtk.vtkImageData()
    vti_image.SetSpacing(patientMeta.Spacing[0], patientMeta.Spacing[1], patientMeta.Spacing[2])
    vti_image.SetOrigin(patientMeta.Origin[0], patientMeta.Origin[1], patientMeta.Origin[2])
    vti_image.SetDimensions(patientMeta.Dimensions[0], patientMeta.Dimensions[1], patientMeta.Dimensions[2])
    return vti_image


def __arr3ToVTS(arr: np.ndarray, patientMeta: PatientMeta, ds: Optional[dicom.Dataset] = None, thisTime: Optional[float]=0.0) -> vtk.vtkStructuredGrid:
    dummyPatientMeta = PatientMeta()
    dummyPatientMeta.setMeta('Dimensions', arr.shape)
    ii = _arrToImagedata(arr, patientMeta=dummyPatientMeta) # No info here, let the transform take care of it
    vts_data = patientMeta.transformVTKData(ii)
    if ds is not None:
        addFieldDataFromDcmDataSet(vts_data, ds, extra_tags={"SliceVector": patientMeta.SliceVector,
                                                                "Time": thisTime})
    return vts_data


def arrToVTS(arr: np.ndarray, 
            patientMeta: PatientMeta, 
            ds: Optional[dicom.Dataset] = None, 
            outputPath: Optional[str] = None) -> Dict[float, vtk.vtkStructuredGrid]:
    dims = arr.shape
    vtkDict = {}
    for k1 in range(dims[-1]):
        A3 = arr[:,:,:,k1]
        A3 = np.swapaxes(A3, 0, 1)
        try:
            thisTime = patientMeta.Times[k1]
        except KeyError:
            thisTime = k1
        vts_data = __arr3ToVTS(A3, patientMeta, ds, thisTime)
        if outputPath is not None:
            vts_data = fIO.writeVTKFile(vts_data, os.path.join(outputPath, f"data_{generate_uid()}_{k1:05d}.vts"))
        vtkDict[thisTime] = vts_data
    return vtkDict


def writeArrToVTI(arr: np.ndarray, patientMeta: PatientMeta, filePrefix: str, outputPath: str, ds: Optional[dicom.Dataset] = None, TRUE_ORIENTATION: bool = False) -> str:
    """Will write a VTI file(s) from arr (if np.ndim(arr)=4 write vti files + pvd file)

    Args:
        arr (np.array): Array of pixel data, shape: nR,nC,nSlice,nTime
        patientMeta (PatientMatrix): PatientMatrix object containing meta to be added as Field data
        filePrefix (str): File name prefix (if nTime>1 then named '{fileprefix}_{timeID:05d}.vti)
        outputPath (str): Output path (if nTime > 1 then '{fileprefix}.pvd written to outputPath and sub-directory holds `*.vti` files)
        ds (pydicom dataset [optional]): pydicom dataset to use to add dicom tags as field data

    Raises:
        ValueError: If VTK import not available
    """
    vtkDict = arrToVTI(arr, patientMeta, ds=ds, TRUE_ORIENTATION=TRUE_ORIENTATION)
    return writeVTIDict(vtkDict, outputPath, filePrefix)

def writeVTIDict(vtiDict: Dict[float, vtk.vtkImageData], outputPath: str, filePrefix: str) -> str:
    times = sorted(vtiDict.keys())
    if len(times) > 1:
        return fIO.writeVTK_PVD_Dict(vtiDict, outputPath, filePrefix, 'vti', BUILD_SUBDIR=True)
    else:
        fOut = os.path.join(outputPath, f'{filePrefix}.vti')
        if type(vtiDict[times[0]]) == str:
            os.rename(vtiDict[times[0]], fOut)
        else:
            fIO.writeVTKFile(vtiDict[times[0]], fOut)
        return fOut 

def scaleVTI(vti_data: vtk.vtkImageData, factor: float) -> None:
    vti_data.SetOrigin([i*factor for i in vti_data.GetOrigin()])
    vti_data.SetSpacing([i*factor for i in vti_data.GetSpacing()])

def filterResampleToImage(vtsObj: vtk.vtkStructuredGrid, target_spacing: List[float]) -> vtk.vtkStructuredGrid:
    rif = vtk.vtkResampleToImage()
    rif.SetInputDataObject(vtsObj)    
    try:
        target_spacing[0]
    except IndexError:
        target_spacing = [target_spacing, target_spacing, target_spacing]
    bounds = vtsObj.GetBounds()
    dims = [
        int((bounds[1] - bounds[0]) / target_spacing[0]),
        int((bounds[3] - bounds[2]) / target_spacing[1]),
        int((bounds[5] - bounds[4]) / target_spacing[2])
    ]
    rif.SetSamplingDimensions(dims[0],dims[1],dims[2])
    rif.Update()
    return rif.GetOutput()


def readImageStackToVTI(imageFileNames: List[str], patientMeta: PatientMeta=None, arrayName: str = 'PixelData', CONVERT_TO_GREYSCALE: bool = False) -> vtk.vtkImageData:
    append_filter = vtk.vtkImageAppend()
    append_filter.SetAppendAxis(2)  # Combine images along the Z axis
    for file_name in imageFileNames:
        thisImage = fIO.readVTKFile(file_name)
        append_filter.AddInputData(thisImage)
    append_filter.Update()
    combinedImage = append_filter.GetOutput()
    if patientMeta is None:
        patientMeta = PatientMeta()
    combinedImage.SetOrigin(patientMeta.Origin)
    combinedImage.SetSpacing(patientMeta.Spacing)
    a = vtkfilters.getScalarsAsNumpy(combinedImage, RETURN_3D=True)
    if CONVERT_TO_GREYSCALE:
        a = np.mean(a, -1)
    elif a.shape[3] == 4: # Remove alpha
        a = a[:,:,:,:3] 
    #
    vtkfilters.setArrayFromNumpy(combinedImage, a, arrayName, IS_3D=True, SET_SCALAR=True)
    vtkfilters.delArraysExcept(combinedImage, [arrayName])
    # Manipulation required to bring jpgs to same orientation as vti
    permute = vtk.vtkImagePermute()
    permute.SetInputData(combinedImage)
    permute.SetFilteredAxes(1, 0, 2)
    permute.Update()
    flip = vtk.vtkImageFlip()
    flip.SetInputData(permute.GetOutput())
    flip.SetFilteredAxis(0)
    flip.Update()
    return flip.GetOutput()


def _process_phase_time_point(args):
    """Helper function for parallel processing of 4D flow time points
    """
    iTime, magFile, phaseFiles_dicts, phase_factors, phase_offsets, scale_factor, velArrayName, rootDir, fName = args
    iVTS = fIO.readVTKFile(magFile)
    thisVelArray = []
    for k2, iPhase in enumerate(phaseFiles_dicts):
        thisPhase_time = ftk.getClosestFloat(iTime, list(iPhase.keys()))
        thisPhase = fIO.readVTKFile(iPhase[thisPhase_time])
        aName = vtkfilters.getScalarsArrayName(thisPhase)
        thisVelArray.append(vtkfilters.getArrayAsNumpy(thisPhase, aName)*phase_factors[k2] + phase_offsets[k2])
    thisVelArray_ = np.array(thisVelArray).T
    thisVelArray_ *= scale_factor
    vtkfilters.setArrayFromNumpy(iVTS, thisVelArray_, velArrayName, SET_VECTOR=True)
    fOutTemp = fIO.writeVTKFile(iVTS, os.path.join(rootDir, f"{fName}_{generate_uid()}.WORKING.vts"))
    return (iTime, fOutTemp)

def mergePhaseSeries4D(magPVD, phasePVD_list, outputFileName, phase_factors, phase_offsets,
                        velArrayName, scale_factor=0.001, DEL_ORIGINAL=True):
    """Take Mag PVD (vts format) and phase PVDs (vts format) and merge into 4D flow dataset

    Args:
        magPVD (str): Path to magnitude PVD file
        phasePVD_list (list): List of paths to phase PVD files
        outputFileName (str): Name of output file
        phase_factors (list): List of factors to multiply phases by. 
        phase_offsets (list): List of offsets to add to phases.
        scale_factor (float): Scale factor to multiply final velocities by. Default is 0.001 (match vel to spatial units)
        velArrayName (str): Name of velocity array to use in output file. Default is "Vels"
        DEL_ORIGINAL (bool, optional): Delete original/intermediate files. Defaults to True.

    Returns:
        str: Name of output file
    """
    rootDir, fName = os.path.split(outputFileName)
    fName = os.path.splitext(fName)[0]
    magFiles = fIO.readPVDFileName(magPVD)
    phaseFiles_dicts = [fIO.readPVDFileName(i) for i in phasePVD_list]
    times = sorted(magFiles.keys())
    outputFilesDict = {}
    # Create process pool and process all time points in parallel
    with Pool() as pool:
        args = [(iTime, magFiles[iTime], phaseFiles_dicts, phase_factors, phase_offsets, 
                scale_factor, velArrayName, rootDir, fName) 
                for iTime in times]
        
        results = pool.map(_process_phase_time_point, args)
    # Collect results into outputFilesDict
    outputFilesDict = dict(results)
    # Generate final output file here - then clean up intermediate files
    pvdFileOut = fIO.writeVTK_PVD_Dict(outputFilesDict, rootDir, fName, "vts", BUILD_SUBDIR=True)

    if DEL_ORIGINAL:
        fIO.deleteFilesByPVD(magPVD)
        for iPhaseFile in phasePVD_list:
            fIO.deleteFilesByPVD(iPhaseFile)
    return pvdFileOut

# ===================================================================================================
# ===================================================================================================
## DICOM-SEG
# ===================================================================================================
def vti_to_dcm_seg(vtiFile, labelMapArrayName, source_dicom_ds_list, dcmSegFileOut=None, algorithm_identification=None):
    imageData = fIO.readVTKFile(vtiFile)
    arr = vtkfilters.getArrayAsNumpy(imageData, labelMapArrayName)
    arr = np.transpose(np.squeeze(arr), axes=[2,0,1])
    return array_to_DcmSeg(arr, source_dicom_ds_list, dcmSegFileOut=dcmSegFileOut, algorithm_identification=algorithm_identification)


def array_to_DcmSeg(arr, source_dicom_ds_list, dcmSegFileOut=None, algorithm_identification=None):
    """Convert a numpy array to DICOM-SEG format.
    
    Args:
        arr: Numpy array of segmentation labels
        source_dicom_ds_list: List of source DICOM datasets
        dcmSegFileOut: Output filename (optional)
        algorithm_identification: Algorithm identification sequence (optional)
        
    Returns:
        DICOM-SEG dataset or filename if dcmSegFileOut is provided
    """

    fullLabelMap = arr.astype(np.ushort)
    sSeg = sorted(set(fullLabelMap.flatten('F')))
    sSeg.remove(0)
    sSegDict = {}
    for k1, segID in enumerate(sSeg):
        sSegDict[k1+1] = f"Segment{k1+1}"
        fullLabelMap[fullLabelMap==segID] = k1+1

    # Describe the algorithm that created the segmentation if not given
    if algorithm_identification is None:
        algorithm_identification = AlgorithmIdentificationSequence(
            name='Spydcmtk',
            version='1.0',
            family=codes.cid7162.ArtificialIntelligence
        )
    segDesc_list = []
    # Describe the segment
    for segID, segName in sSegDict.items():
        description_segment = SegmentDescription(
            segment_number=segID,
            segment_label=segName,
            segmented_property_category=codes.cid7150.Tissue,
            segmented_property_type=codes.cid7154.Kidney,
            algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm_identification,
            tracking_uid=generate_uid(),
            tracking_id='spydcmtk %s'%(segName)
        )
        segDesc_list.append(description_segment)
    
    try:
        # Create the Segmentation instance
        seg_dataset = Segmentation(
            source_images=source_dicom_ds_list,
            pixel_array=fullLabelMap,
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=segDesc_list,
            series_instance_uid=generate_uid(),
            series_number=2,
            sop_instance_uid=generate_uid(),
            instance_number=1,
            manufacturer='Manufacturer',
            manufacturer_model_name='Model',
            software_versions='v1',
            device_serial_number='Device XYZ',
        )
    except Exception as e:
        print(f"Error creating segmentation: {str(e)}")
        print(f"Number of source images: {len(source_dicom_ds_list)}")
        print(f"Label map shape: {fullLabelMap.shape}")
        raise

    if dcmSegFileOut is not None:
        seg_dataset.save_as(dcmSegFileOut) 
        return dcmSegFileOut
    return seg_dataset


def getDcmSeg_meta_depreciated(dcmseg):
    sliceThick = dcmseg.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness
    pixSpace = dcmseg.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
    ffgs = dcmseg.PerFrameFunctionalGroupsSequence
    ipp = [i.PlanePositionSequence[0].ImagePositionPatient for i in dcmseg.PerFrameFunctionalGroupsSequence]
    oo = np.array(ipp[0])
    normalVector = np.array(ipp[-1]) - oo 
    normalVector = normalVector / np.linalg.norm(normalVector)
    oo = dcmseg.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient
    iop = dcmseg.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient
    seg_data = dcmseg.pixel_array
    seg_data = np.transpose(seg_data, axes=[2,1,0])
    return {"Origin": [oo[0]*0.001, oo[1]*0.001, oo[2]*0.001], 
            "Spacing": [pixSpace[0]*0.001, pixSpace[1]*0.001, sliceThick*0.001],
            "Dimensions": seg_data.shape,
            "ImageOrientationPatient": iop, 
            "SliceVector": normalVector   
            }


# def dicom_seg_to_vtk_depreciated(dicom_seg_path, vtk_output_path, TRUE_ORIENTATION=False):
#     ds = dicom.dcmread(dicom_seg_path)
#     patMeta = PatientMeta()
#     patMeta.initFromDicomSeg(ds)
#     seg_data = ds.pixel_array
#     seg_data = np.transpose(seg_data, axes=[2,1,0])
#     image_data = vtk.vtkImageData()
#     image_data.SetOrigin(patMeta.Origin)
#     image_data.SetDimensions(patMeta.Dimensions)
#     image_data.SetSpacing(patMeta.Spacing)
#     vtk_array = numpy_support.numpy_to_vtk(num_array=seg_data.flatten('F'), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
#     image_data.GetPointData().SetScalars(vtk_array)
#     if TRUE_ORIENTATION:
#         fIO.writeVTS(_vti2vts(image_data, patMeta), vtk_output_path)
#     else:
#         fIO.writeVTI(image_data, vtk_output_path)
#     return vtk_output_path


class NoVtkError(Exception):
    ''' NoVtkError
            If VTK import fails '''
    def __init__(self):
        pass
    def __str__(self):
        return 'NoVtkError: VTK not found. Run: "pip install vtk"'


# ===================================================================================================
# ===================================================================================================



# ===================================================================================================
# ===================================================================================================
## HELPFUL FILTERS
# ===================================================================================================

def addFieldDataFromDcmDataSet(vtkObj, ds, extra_tags={}):
    tagsDict = dcmTools.getDicomTagsDict()
    for iTag in tagsDict.keys():
        try:
            val = ds[iTag].value
            if type(val) in [dicom.multival.MultiValue, dicom.valuerep.DSfloat, dicom.valuerep.IS]:
                try:
                    tagArray = numpy_support.numpy_to_vtk(np.array(val))
                except TypeError: # multivalue - but prob strings
                    tagArray = vtk.vtkStringArray()
                    tagArray.SetNumberOfValues(len(val))
                    for k1 in range(len(val)):
                        tagArray.SetValue(k1, str(val[k1]))
            else:
                tagArray = vtk.vtkStringArray()
                tagArray.SetNumberOfValues(1)
                tagArray.SetValue(0, str(val))
            tagArray.SetName(iTag)
            vtkObj.GetFieldData().AddArray(tagArray)
        except KeyError:
            continue # tag not found
    for iTag in extra_tags:
        val = extra_tags[iTag]
        tagArray = numpy_support.numpy_to_vtk(np.array(val))
        tagArray.SetName(iTag)
        vtkObj.GetFieldData().AddArray(tagArray)
