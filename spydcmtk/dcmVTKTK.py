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
try:
    from highdicom.seg.content import SegmentDescription
    from highdicom.seg.enum import SegmentAlgorithmTypeValues, SegmentationTypeValues
    from highdicom.content import AlgorithmIdentificationSequence
    from highdicom.seg.sop import Segmentation
    HIGHDCM = True
except ImportError:
    HIGHDCM = False

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import vtk
from vtk.util import numpy_support # type: ignore

import spydcmtk.dcmTools as dcmTools




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
        if 'SliceThickness' in self._meta:
            return self._meta['PixelSpacing'][0], self._meta['PixelSpacing'][1], self._meta['SliceThickness']
        else:
            return self._meta['PixelSpacing'][0], self._meta['PixelSpacing'][1], self._meta['SpacingBetweenSlices']
    
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
            if 'SliceThickness' not in self._meta:
                self._meta['SliceThickness'] = self._meta['Spacing'][2]
        if 'SpacingBetweenSlices' not in self._meta:
            self._meta['SpacingBetweenSlices'] = self._meta['SliceThickness']
        if 'Origin' in self._meta:
            self._meta['ImagePositionPatient'] = [self._meta['Origin'][0], self._meta['Origin'][1], self._meta['Origin'][2]]

    def initFromDicomSeries(self, dicomSeries):
        I,J,K = int(dicomSeries.getTag('Rows')), int(dicomSeries.getTag('Columns')), int(dicomSeries.getNumberOfSlicesPerVolume())
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
        ipp = dicomSeries.getImagePositionPatient_np(0)
        oo = [i*0.001 for i in ipp]
        sliceVec = dicomSeries.getSliceNormalVector()
        self._meta = {
                    'PixelSpacing': [dicomSeries.getDeltaCol()*0.001, dicomSeries.getDeltaRow()*0.001],
                    'SpacingBetweenSlices': dicomSeries.getDeltaSlice()*0.001,
                    'SliceThickness': dicomSeries.getTag('SliceThickness', convertToType=float, ifNotFound=dicomSeries.getDeltaSlice())*0.001,
                    'SliceLocation0': dicomSeries.getTag('SliceLocation', 0, ifNotFound=0.0, convertToType=float)*0.001,
                    'ImagePositionPatient': oo, 
                    'ImageOrientationPatient': dicomSeries.getTag('ImageOrientationPatient'), 
                    'PatientPosition': dicomSeries.getTag('PatientPosition'), 
                    'Times': [dt*n*0.001 for n in range(N)],
                    'Dimensions': A.shape,
                    'SliceVector': sliceVec,
                }
        self._updateMatrix()

    def initFromVTI(self, vtiObj, scaleFactor=1.0):
        dx,dy,dz = vtiObj.GetSpacing()
        oo = vtiObj.GetOrigin()
        # 1st option from meta, then field data then default
        iop = getFieldData(vtiObj, 
                            'ImageOrientationPatient', 
                            default=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        sliceVec = getFieldData(vtiObj, 
                                'SliceVector', 
                                default=[0.0, 0.0, 1.0])
        self._meta = {'PixelSpacing': [dx*scaleFactor, dy*scaleFactor],
                            'ImagePositionPatient': [i*scaleFactor for i in oo],
                            'ImageOrientationPatient': iop,
                            'SpacingBetweenSlices': dz*scaleFactor,
                            'SliceVector': sliceVec,
                            'Dimensions': vtiObj.GetDimensions(),
                            'SliceThickness': dz*scaleFactor,
                            'SliceLocation0': 0.0}
        self._updateMatrix()

    def initFromDicomSeg(self, dicomSeg, scaleFactor=1.0):
        sliceThick = dicomSeg.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness
        pixSpace = dicomSeg.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        ipp = [i.PlanePositionSequence[0].ImagePositionPatient for i in dicomSeg.PerFrameFunctionalGroupsSequence]
        oo = np.array(ipp[0])
        normalVector = np.array(ipp[-1]) - oo 
        normalVector = normalVector / np.linalg.norm(normalVector)
        oo = dicomSeg.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient
        iop = dicomSeg.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient
        seg_data = dicomSeg.pixel_array
        seg_data = np.transpose(seg_data, axes=[2,1,0])
        self._meta = {"ImagePositionPatient": [oo[0]*scaleFactor, oo[1]*scaleFactor, oo[2]*scaleFactor], 
                "PixelSpacing": [pixSpace[0]*scaleFactor, pixSpace[1]*scaleFactor],
                "SliceThickness": sliceThick*scaleFactor,
                "SpacingBetweenSlices": sliceThick*scaleFactor,
                "Dimensions": seg_data.shape,
                "ImageOrientationPatient": iop, 
                "SliceVector": normalVector   
                }
        self._updateMatrix()

    def _updateMatrix(self):
        self._matrix = self.buildImage2PatientCoordinateMatrix()

    def buildImage2PatientCoordinateMatrix(self):
        dx, dy, dz = self.Spacing
        oo = self.ImagePositionPatient
        orientation = np.array(self.ImageOrientationPatient)
        iop = np.vstack((orientation.reshape(2, 3), self.SliceVector))
        matrix = np.array([
            [iop[0,0]*dx, iop[0,1]*dy, iop[0,2]*dz, oo[0]], 
            [iop[1,0]*dx, iop[1,1]*dy, iop[1,2]*dz, oo[1]], 
            [iop[2,0]*dx, iop[2,1]*dy, iop[2,2]*dz, oo[2]], 
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
            'ImagePositionPatient': np.array([i*1000.0 for i in self.ImagePositionPatient]),
            'PixelSpacing': np.array([i*1000.0 for i in self.PixelSpacing]),
            'ImageOrientationPatient': self.ImageOrientationPatient,
            'SliceVector': self.SliceVector,
            'SliceThickness': self.SliceThickness*1000.0,
            'SliceLocation0': self.SliceLocation0*1000.0,
            'SpacingBetweenSlices': self.SpacingBetweenSlices*1000.0
        }   

    def imageToPatientCoordinates(self, imageCoords):
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

def arrToVTI(arr: np.ndarray, patientMeta: PatientMeta, ds: Optional[dicom.Dataset] = None, TRUE_ORIENTATION: bool = False) -> Dict[float, vtk.vtkImageData]:
    """Convert array (+meta) to VTI dict (keys=times, values=VTI volumes). 

    Args:
        arr (np.array): Array of pixel data, shape: nR,nC,nSlice,nTime
        patientMeta (PatientMatrix): PatientMatrix object containing meta to be added as Field data
        ds (pydicom dataset [optional]): pydicom dataset to use to add dicom tags as field data
        TRUE_ORIENTATION (bool [False]) : Boolean to force accurate spatial location of image data.
                                NOTE: this uses resampling from VTS data so output VTI will have different dimensions.  
    
    Returns:
        vtiDict

    Raises:
        ValueError: If VTK import not available
    """
    dims = arr.shape
    vtkDict = {}
    timesUsed = []
    for k1 in range(dims[-1]):
        A3 = arr[:,:,:,k1]
        ###
        A3 = np.swapaxes(A3, 0, 1)
        newImg = _arrToImagedata(A3, patientMeta)
        if TRUE_ORIENTATION:
            vts_data = _vti2vts(newImg, patientMeta)
            newImg = filterResampleToImage(vts_data, np.min(patientMeta.Spacing))
            delAllCellArrays(newImg)
            delArraysExcept(newImg, ['PixelData'])
        if ds is not None:
            addFieldDataFromDcmDataSet(newImg, ds, extra_tags={"SliceVector": patientMeta.SliceVector,
                                                                "Time": patientMeta.Times[0]})
        try:
            thisTime = patientMeta.Times[k1]
        except KeyError:
            thisTime = k1
        if thisTime in timesUsed:
            thisTime = k1
        timesUsed.append(thisTime)
        vtkDict[thisTime] = newImg
    return vtkDict

def _arrToImagedata(A3: np.ndarray, patientMeta: PatientMeta) -> vtk.vtkImageData:
    newImg = _buildVTIImage(patientMeta)
    npArray = np.reshape(A3, np.prod(A3.shape), 'F').astype(np.int16)
    aArray = numpy_support.numpy_to_vtk(npArray, deep=1)
    aArray.SetName('PixelData')
    newImg.GetPointData().SetScalars(aArray)
    return newImg

def _vti2vts(vti_image: vtk.vtkImageData, patientMeta: PatientMeta) -> vtk.vtkStructuredGrid:
    vti_image.SetOrigin(0.0,0.0,0.0) # Origin should be in the patientMeta
    return patientMeta.transformVTKData(vti_image)
    
def _buildVTIImage(patientMeta: PatientMeta=None) -> vtk.vtkImageData:
    if patientMeta is None:
        patientMeta = PatientMeta()
    vti_image = vtk.vtkImageData()
    vti_image.SetSpacing(patientMeta.Spacing[0] ,patientMeta.Spacing[1] ,patientMeta.Spacing[2])
    vti_image.SetOrigin(patientMeta.Origin[0], patientMeta.Origin[1], patientMeta.Origin[2])
    vti_image.SetDimensions(patientMeta.Dimensions[1], patientMeta.Dimensions[0], patientMeta.Dimensions[2])
    return vti_image

def arrToVTS(arr: np.ndarray, patientMeta: PatientMeta, ds: Optional[dicom.Dataset] = None) -> Dict[float, vtk.vtkStructuredGrid]:
    dims = arr.shape
    vtkDict = {}
    timesUsed = []
    for k1 in range(dims[-1]):
        A3 = arr[:,:,:,k1]
        A3 = np.swapaxes(A3, 0, 1)
        ii = _arrToImagedata(A3, patientMeta)
        vts_data = _vti2vts(ii, patientMeta)
        if ds is not None:
            addFieldDataFromDcmDataSet(vts_data, ds, extra_tags={"SliceVector": patientMeta.SliceVector,
                                                                "Time": patientMeta.Times[0]})
        try:
            thisTime = patientMeta.Times[k1]
        except KeyError:
            thisTime = k1
        if thisTime in timesUsed:
            thisTime = k1
        timesUsed.append(thisTime)
        vtkDict[thisTime] = vts_data
    return vtkDict

def writeArrToVTI(arr: np.ndarray, patientMeta: PatientMeta, filePrefix: str, outputPath: str, ds: Optional[dicom.Dataset] = None, TRUE_ORIENTATION: bool = False) -> str:
    """Will write a VTI file(s) from arr (if np.ndim(arr)=4 write vti files + pvd file)

    Args:
        arr (np.array): Array of pixel data, shape: nR,nC,nSlice,nTime
        patientMeta (PatientMatrix): PatientMatrix object containing meta to be added as Field data
        filePrefix (str): File name prefix (if nTime>1 then named '{fileprefix}_{timeID:05d}.vti)
        outputPath (str): Output path (if nTime > 1 then '{fileprefix}.pvd written to outputPath and sub-directory holds *.vti files)
        ds (pydicom dataset [optional]): pydicom dataset to use to add dicom tags as field data

    Raises:
        ValueError: If VTK import not available
    """
    vtkDict = arrToVTI(arr, patientMeta, ds=ds, TRUE_ORIENTATION=TRUE_ORIENTATION)
    return writeVTIDict(vtkDict, outputPath, filePrefix)

def writeVTIDict(vtiDict: Dict[float, vtk.vtkImageData], outputPath: str, filePrefix: str) -> str:
    times = sorted(vtiDict.keys())
    if len(times) > 1:
        return writeVtkPvdDict(vtiDict, outputPath, filePrefix, 'vti', BUILD_SUBDIR=True)
    else:
        fOut = os.path.join(outputPath, f'{filePrefix}.vti')
        return writeVTI(vtiDict[times[0]], fOut)

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
        thisImage = readVTKFile(file_name)
        append_filter.AddInputData(thisImage)
    append_filter.Update()
    combinedImage = append_filter.GetOutput()
    if patientMeta is None:
        patientMeta = PatientMeta()
    combinedImage.SetOrigin(patientMeta.Origin)
    combinedImage.SetSpacing(patientMeta.Spacing)
    a = getScalarsAsNumpy(combinedImage)
    if CONVERT_TO_GREYSCALE:
        a = np.mean(a, 1)
    addArrayFromNumpy(combinedImage, a, arrayName, SET_SCALAR=True)
    delArraysExcept(combinedImage, [arrayName])
    return combinedImage


# =========================================================================
# =========================================================================
## DICOM-SEG
# =========================================================================
def array_to_DcmSeg(arr, source_dicom_ds_list, dcmSegFileOut=None, algorithm_identification=None):
    if not HIGHDCM:
        raise ImportError("Missing highdicom \n Please run: pip install highdicom")
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
    # Create the Segmentation instance
    seg_dataset = Segmentation(
        source_images=source_dicom_ds_list,
        pixel_array=fullLabelMap,
        segmentation_type=SegmentationTypeValues.BINARY,
        segment_descriptions=segDesc_list,
        series_instance_uid=generate_uid(), #source_dicom_ds_list[0].SeriesInstanceUID,
        series_number=2,
        sop_instance_uid=generate_uid(), #source_dicom_ds_list[0].SOPInstanceUID,
        instance_number=1,
        manufacturer='Manufacturer',
        manufacturer_model_name='Model',
        software_versions='v1',
        device_serial_number='Device XYZ',
    )
    if dcmSegFileOut is not None:
        seg_dataset.save_as(dcmSegFileOut) 
        return dcmSegFileOut
    return seg_dataset


def getDcmSeg_meta(dcmseg):
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


def dicom_seg_to_vtk(dicom_seg_path, vtk_output_path, TRUE_ORIENTATION=False):
    ds = dicom.dcmread(dicom_seg_path)
    patMeta = PatientMeta()
    patMeta.initFromDicomSeg(ds)
    seg_data = ds.pixel_array
    seg_data = np.transpose(seg_data, axes=[2,1,0])
    image_data = vtk.vtkImageData()
    image_data.SetOrigin(patMeta.Origin)
    image_data.SetDimensions(patMeta.Dimensions)
    image_data.SetSpacing(patMeta.Spacing)
    vtk_array = numpy_support.numpy_to_vtk(num_array=seg_data.flatten('F'), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    image_data.GetPointData().SetScalars(vtk_array)
    if TRUE_ORIENTATION:
        writeVTS(_vti2vts(image_data, patMeta), vtk_output_path)
    else:
        writeVTI(image_data, vtk_output_path)
    return vtk_output_path


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
## TO REMOVE TO NGAWARI
# ===================================================================================================
def __writerWrite(writer, data, fileName: str) -> str:
    writer.SetFileName(fileName)
    writer.SetInputData(data)
    writer.Write()
    return fileName


def writeNII(data: vtk.vtkImageData, fileName: str) -> str:
    writer = vtk.vtkNIFTIImageWriter()
    return __writerWrite(writer, data, fileName)


def writeMHA(data: vtk.vtkImageData, fileName: str) -> str:
    writer = vtk.vtkMetaImageWriter()
    return __writerWrite(writer, data, fileName)


def writeVTS(data: vtk.vtkStructuredGrid, fileName: str) -> str:
    writer = vtk.vtkXMLStructuredGridWriter()
    return __writerWrite(writer, data, fileName)


def writeVTI(data: vtk.vtkImageData, fileName: str) -> str:
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetDataModeToBinary()
    return __writerWrite(writer, data, fileName)


def nii2vti(fullFileName: str) -> vtk.vtkImageData:
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(fullFileName)
    reader.Update()
    data = reader.GetOutput()
    ## TRANSFORM
    qFormMatrix = reader.GetQFormMatrix()
    trans = vtk.vtkTransform()
    trans.SetMatrix(qFormMatrix)
    transFilter = vtk.vtkTransformFilter()
    transFilter.SetTransform(trans)
    transFilter.SetInputData(data)
    transFilter.Update()
    dataT = transFilter.GetOutput()
    ## RESAMPLE BACK TO VTI
    rif = vtk.vtkResampleToImage()
    rif.SetInputDataObject(dataT)
    d1,d2,d3 = dataT.GetDimensions()
    rif.SetSamplingDimensions(d1,d2,d3)
    rif.Update()
    data = rif.GetOutput()
    ## WRITE
    dd, ff = os.path.split(fullFileName)
    ff, _ = os.path.splitext(ff)
    fOut = os.path.join(dd, ff+'.vti')
    writeVTI(data, fOut)
    return fOut

def writeVtkFile(data: vtk.vtkImageData, fileName: str) -> str:
    if fileName.endswith('.vti'):
        return writeVTI(data, fileName)
    elif fileName.endswith('.vts'):
        return writeVTS(data, fileName)
    elif fileName.endswith('.mha'):
        return writeMHA(data, fileName)
    
def readVTKFile(fileName: str) -> vtk.vtkImageData:
    # --- CHECK EXTENSION - READ FILE ---
    if not os.path.isfile(fileName):
        raise IOError('## ERROR: %s file not found'%(fileName))
    if fileName.endswith('vtp'):
        reader = vtk.vtkXMLPolyDataReader()
    elif fileName.endswith('vts'):
        reader = vtk.vtkXMLStructuredGridReader()
    elif fileName.endswith('vtu'):
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif fileName.endswith('stl'):
        reader = vtk.vtkSTLReader()
        reader.ScalarTagsOn()
    elif fileName.endswith('nii'):
        reader = vtk.vtkNIFTIImageReader()
    elif fileName.endswith('vti'):
        reader = vtk.vtkXMLImageDataReader()
    elif fileName.endswith('vtk'):
        reader = vtk.vtkPolyDataReader()
    elif fileName.endswith('vtm'):
        reader = vtk.vtkXMLMultiBlockDataReader()
    elif fileName.endswith('nrrd'):
        reader = vtk.vtkNrrdReader()
    elif fileName.endswith('mha') | fileName.endswith('mhd'):
        reader = vtk.vtkMetaImageReader()
    elif fileName.endswith('png'):
        reader = vtk.vtkPNGReader()
    elif fileName.endswith('jpg'):
        reader = vtk.vtkJPEGReader()
    elif fileName.endswith('pvd'):
        raise IOError(' PVD - should use readPVD()')
    else:
        raise IOError(fileName + ' not correct extension')
    reader.SetFileName(fileName)
    reader.Update()
    return reader.GetOutput()

# =========================================================================
##          PVD Stuff
# =========================================================================
def checkIfExtnPresent(fileName: str, extn: str) -> str:
    if (extn[0] == '.'):
        extn = extn[1:]
    le = len(extn)
    if (fileName[-le:] != extn):
        fileName = fileName + '.' + extn
    return fileName

def _writePVD(rootDirectory: str, filePrefix: str, outputSummary: Dict[int, Dict[str, Any]]) -> str:
    """
    :param rootDirectory:
    :param filePrefix:
    :param outputSummary: dict of dicts : { timeID : {TrueTime : float, FileName : str}
    :return: full file name
    """
    fileOut = os.path.join(rootDirectory, filePrefix + '.pvd')
    with open(fileOut, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('<Collection>\n')
        for timeId in sorted(outputSummary.keys()):
            sTrueTime = outputSummary[timeId]['TrueTime']
            tFileName = str(outputSummary[timeId]['FileName'])
            f.write('<DataSet timestep="%7.5f" file="%s"/>\n' % (sTrueTime, tFileName))
        f.write('</Collection>\n')
        f.write('</VTKFile>')
    return fileOut


def _makePvdOutputDict(vtkDict: Dict[int, vtk.vtkImageData], filePrefix: str, fileExtn: str, subDir: str = '') -> Dict[int, Dict[str, Any]]:
    outputSummary = {}
    myKeys = vtkDict.keys()
    myKeys = sorted(myKeys)
    for timeId in range(len(myKeys)):
        fileName = __buildFileName(filePrefix, timeId, fileExtn)
        trueTime = myKeys[timeId]
        outputMeta = {'FileName': os.path.join(subDir, fileName), 'TimeID': timeId, 'TrueTime': trueTime}
        outputSummary[timeId] = outputMeta
    return outputSummary

def __writePvdData(vtkDict: Dict[int, vtk.vtkImageData], rootDir: str, filePrefix: str, fileExtn: str, subDir: str = '') -> None:
    myKeys = vtkDict.keys()
    myKeys = sorted(myKeys)
    for timeId in range(len(myKeys)):
        fileName = __buildFileName(filePrefix, timeId, fileExtn)
        fileOut = os.path.join(rootDir, subDir, fileName)
        if type(vtkDict[myKeys[timeId]]) == str:
            os.rename(vtkDict[myKeys[timeId]], fileOut)
        else:
            writeVtkFile(vtkDict[myKeys[timeId]], fileOut)

def writeVtkPvdDict(vtkDict: Dict[int, vtk.vtkImageData], rootDir: str, filePrefix: str, fileExtn: str, BUILD_SUBDIR: bool = True) -> str:
    """
    Write dict of time:vtkObj to pvd file
        If dict is time:fileName then will copy files
    :param vtkDict: python dict - time:vtkObj
    :param rootDir: directory
    :param filePrefix: make filePrefix.pvd
    :param fileExtn: file extension (e.g. vtp, vti, vts etc)
    :param BUILD_SUBDIR: bool - to build subdir (filePrefix.pvd in root, then data in root/filePrefix/
    :return: full file name
    """
    filePrefix = os.path.splitext(filePrefix)[0]
    subDir = ''
    fullPVD = os.path.join(rootDir, checkIfExtnPresent(filePrefix, 'pvd'))
    if os.path.isfile(fullPVD) & (type(list(vtkDict.values())[0]) != str):
        deleteFilesByPVD(fullPVD, QUIET=True)
    if BUILD_SUBDIR:
        subDir = filePrefix
        if not os.path.isdir(os.path.join(rootDir, subDir)):
            os.mkdir(os.path.join(rootDir, subDir))
    outputSummary = _makePvdOutputDict(vtkDict, filePrefix, fileExtn, subDir)
    __writePvdData(vtkDict, rootDir, filePrefix, fileExtn, subDir)
    return _writePVD(rootDir, filePrefix, outputSummary)

def deleteFilesByPVD(pvdFile: str, FILE_ONLY: bool = False, QUIET: bool = False) -> int:
    """
    Will Read pvdFile - delete all files from hard drive that pvd refs
        Then delete pvdFile
    :param pvdFile:
    :param FILE_ONLY:
    :param QUIET:
    :return:
    """
    if FILE_ONLY:
        try:
            os.remove(pvdFile)
        except (IOError, OSError):
            print('    warning - file not found %s' % (pvdFile))
            return 1
        return 0
    try:
        pvdDict = readPVDFileName(pvdFile)
        for iKey in pvdDict.keys():
            os.remove(pvdDict[iKey])
            try:
                os.remove(pvdDict[iKey])
            except OSError:
                pass  # ignore this as may be shared by and deleted by another pvd
        os.remove(pvdFile)
    except (IOError, OSError):
        if (not QUIET)&("pvd" not in pvdFile):
            print('    warning - file not found %s' % (pvdFile))
    try:
        head, _ = os.path.splitext(pvdFile)
        os.rmdir(head)
    except (IOError, OSError):
        if not QUIET:
            print('    warning - dir not found %s' % (os.path.splitext(pvdFile)[0]))
    return 0

def __buildFileName(prefix: str, idNumber: int, extn: str) -> str:
    ids = '%05d'%(idNumber)
    if extn[0] != '.':
        extn = '.' + extn
    fileName = prefix + '_' + ids + extn
    return fileName

def readPVDFileName(fileIn: str, vtpTime: float = 0.0, timeIDs: List[int] = None, RETURN_OBJECTS_DICT: bool = False) -> Dict[float, str]:
    """
    Read PVD file, return dictionary of fullFileNames - keys = time
    So DOES NOT read file
    If not pvd - will return dict of {0.0 : fileName}
    """
    if timeIDs is None:
        timeIDs = []
    _, ext = os.path.splitext(fileIn)
    if ext != '.pvd':
        if RETURN_OBJECTS_DICT:
            return {vtpTime: readVTKFile(fileIn)}
        else:
            return {vtpTime: fileIn}
    #
    vtkDict = pvdGetDict(fileIn, timeIDs)
    if RETURN_OBJECTS_DICT:
        kk = vtkDict.keys()
        return dict(zip(kk, [readVTKFile(vtkDict[i]) for i in kk]))
    else:
        return vtkDict

def readPVD(fileIn: str, timeIDs: List[int] = None) -> Dict[float, str]:
    if timeIDs is None:
        timeIDs = []
    return readPVDFileName(fileIn, timeIDs=timeIDs, RETURN_OBJECTS_DICT=True)

def pvdGetDict(pvd: str, timeIDs: List[int] = None) -> Dict[float, str]:
    if timeIDs is None:
        timeIDs = []
    if type(pvd) == str:
        root = ET.parse(pvd).getroot()
    elif type(pvd) == dict:
        return pvd
    else:
        root = pvd
    nTSteps = len(root[0])
    if len(timeIDs) == 0:
        timeIDs = range(nTSteps)
    else:
        for k1 in range(len(timeIDs)):
            if timeIDs[k1] < 0:
                timeIDs[k1] = nTSteps + timeIDs[k1]
    pvdTimesFilesDict = {}
    rootDir = os.path.dirname(pvd)
    for k in range(nTSteps):
        if k not in timeIDs:
            continue
        a = root[0][k].attrib
        fullVtkFileName = os.path.join(rootDir, a['file'])
        pvdTimesFilesDict[float(a['timestep'])] = fullVtkFileName
    return pvdTimesFilesDict



# =========================================================================
# =========================================================================
## HELPFUL FILTERS
# =========================================================================
def vtkfilterFlipImageData(vtiObj, axis):
    flipper = vtk.vtkImageFlip()
    flipper.SetFilteredAxes(axis)
    flipper.SetInputData(vtiObj)
    flipper.Update()
    return flipper.GetOutput()


def getScalarsAsNumpy(data):
    aS = data.GetPointData().GetScalars()
    aName = aS.GetName()
    return getArrayAsNumpy(data, aName)


def getArrayAsNumpy(data, arrayName):
    return numpy_support.vtk_to_numpy(data.GetPointData().GetArray(arrayName)).copy()


def addArrayFromNumpy(data, npArray, arrayName, SET_SCALAR=False):
    aArray = numpy_support.numpy_to_vtk(npArray, deep=1)
    aArray.SetName(arrayName)
    if SET_SCALAR:
        data.GetPointData().SetScalars(aArray)
    else:
        data.GetPointData().AddArray(aArray)


def addFieldData(vtkObj, fieldVal, fieldName):
    tagArray = numpy_support.numpy_to_vtk(np.array([float(fieldVal)]))
    tagArray.SetName(fieldName)
    vtkObj.GetFieldData().AddArray(tagArray)


def getFieldData(vtkObj, fieldName, default=None):
    try:
        return numpy_support.vtk_to_numpy(vtkObj.GetFieldData().GetArray(fieldName)).copy()
    except AttributeError:
        return default


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


def delArray(data, arrayName):
    data.GetPointData().RemoveArray(arrayName)


def delArraysExcept(data, arrayNamesToKeep_list):
    aList = [data.GetPointData().GetArrayName(i) for i in range(data.GetPointData().GetNumberOfArrays())]
    for ia in aList:
        if ia not in arrayNamesToKeep_list:
            data.GetPointData().RemoveArray(ia)
    return data


def delAllCellArrays(data):
    for i in range(data.GetCellData().GetNumberOfArrays()):
        data.GetCellData().RemoveArray(data.GetCellData().GetArrayName(i))
