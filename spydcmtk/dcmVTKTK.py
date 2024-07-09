"""
Created on MArch 2023 (rewrite from old module - remove reliance on VTKDICOM)

@author: fraser

Dicom to VTK conversion toolkit

"""

import os
import numpy as np
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


# ===================================================================================================
# EXPOSED METHODS
# ===================================================================================================

def arrToVTI(arr, meta, ds=None, TRUE_ORIENTATION=False):
    """Convert array (+meta) to VTI dict (keys=times, values=VTI volumes). 

    Args:
        arr (np.array): Array of pixel data, shape: nR,nC,nSlice,nTime
        meta (dict): dictionary containing meta to be added as Field data
            meta = {'Spacing': list_3 -> resolution, 
                    'Origin': list_3 -> origin, 
                    'ImageOrientationPatient': list_6 -> ImageOrientationPatient, 
                    'Times': list_nTime -> times (can be missing if nTime=1)}
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
        newImg = _arrToImagedata(A3, meta)
        if TRUE_ORIENTATION:
            vts_data = _vti2vts(newImg, meta)
            newImg = filterResampleToImage(vts_data, np.min(meta['Spacing']))
            delAllCellArrays(newImg)
            delArraysExcept(newImg, ['PixelData'])
        if ds is not None:
            addFieldDataFromDcmDataSet(newImg, ds, extra_tags={"SliceVector": meta['SliceVector'],
                                                                "Time": meta.get('Times', [0.0])[0]})
        try:
            thisTime = meta['Times'][k1]
        except KeyError:
            thisTime = k1
        if thisTime in timesUsed:
            thisTime = k1
        timesUsed.append(thisTime)
        vtkDict[thisTime] = newImg
    return vtkDict

def _arrToImagedata(A3, meta):
    newImg = _buildVTIImage(meta)
    npArray = np.reshape(A3, np.prod(A3.shape), 'F').astype(np.int16)
    aArray = numpy_support.numpy_to_vtk(npArray, deep=1)
    aArray.SetName('PixelData')
    newImg.GetPointData().SetScalars(aArray)
    return newImg

def _vti2vts(vti_image, meta):
    return vtiToVts_viaTransform(vti_image, build_vtkTransform_meta(meta))
    
def _buildVTIImage(meta):
    vti_image = vtk.vtkImageData()
    vti_image.SetSpacing(meta['Spacing'][0] ,meta['Spacing'][1] ,meta['Spacing'][2])
    vti_image.SetOrigin(meta['Origin'][0], meta['Origin'][1], meta['Origin'][2])
    vti_image.SetDimensions(meta['Dimensions'][1], meta['Dimensions'][0], meta['Dimensions'][2])
    return vti_image

def _buildVTSGrid_xform(meta): # VTK IMAGE + TRANSFORM (faster than matrix calc)
    vti_image = _buildVTIImage(meta)
    return _vti2vts(vti_image, meta)
    

def arrToVTS(arr, meta, ds=None):
    dims = arr.shape
    vtkDict = {}
    timesUsed = []
    for k1 in range(dims[-1]):
        A3 = arr[:,:,:,k1]
        A3 = np.swapaxes(A3, 0, 1)
        ii = _arrToImagedata(A3, meta)
        vts_data = _vti2vts(ii, meta)
        if ds is not None:
            addFieldDataFromDcmDataSet(vts_data, ds, extra_tags={"SliceVector": meta['SliceVector'],
                                                                "Time": meta.get('Times', [0.0])[0]})
        try:
            thisTime = meta['Times'][k1]
        except KeyError:
            thisTime = k1
        if thisTime in timesUsed:
            thisTime = k1
        timesUsed.append(thisTime)
        vtkDict[thisTime] = vts_data
    return vtkDict

def writeArrToVTI(arr, meta, filePrefix, outputPath, ds=None, TRUE_ORIENTATION=False):
    """Will write a VTI file(s) from arr (if np.ndim(arr)=4 write vti files + pvd file)

    Args:
        arr (np.array): Array of pixel data, shape: nR,nC,nSlice,nTime
        meta (dict): dictionary containing meta to be added as Field data
            meta = {'Spacing': list_3 -> resolution, 
                    'Origin': list_3 -> origin, 
                    'ImageOrientationPatient': list_6 -> ImageOrientationPatient, 
                    'Times': list_nTime -> times (can be missing if nTime=1)}
        filePrefix (str): File name prefix (if nTime>1 then named '{fileprefix}_{timeID:05d}.vti)
        outputPath (str): Output path (if nTime > 1 then '{fileprefix}.pvd written to outputPath and sub-directory holds *.vti files)
        ds (pydicom dataset [optional]): pydicom dataset to use to add dicom tags as field data

    Raises:
        ValueError: If VTK import not available
    """
    vtkDict = arrToVTI(arr, meta, ds=ds, TRUE_ORIENTATION=TRUE_ORIENTATION)
    return writeVTIDict(vtkDict, outputPath, filePrefix)

def writeVTIDict(vtiDict, outputPath, filePrefix):
    times = sorted(vtiDict.keys())
    if len(times) > 1:
        return writeVtkPvdDict(vtiDict, outputPath, filePrefix, 'vti', BUILD_SUBDIR=True)
    else:
        fOut = os.path.join(outputPath, f'{filePrefix}.vti')
        return writeVTI(vtiDict[times[0]], fOut)

def scaleVTI(vti_data, factor):
    vti_data.SetOrigin([i*factor for i in vti_data.GetOrigin()])
    vti_data.SetSpacing([i*factor for i in vti_data.GetSpacing()])


def build_vtkTransform_fieldData(vtkObj):
    iop = [vtkObj.GetFieldData().GetArray('ImageOrientationPatient').GetTuple(i)[0] for i in range(6)]
    SliceVector = [vtkObj.GetFieldData().GetArray('SliceVector').GetTuple(i)[0] for i in range(3)]
    oo = [vtkObj.GetFieldData().GetArray('ImagePositionPatient').GetTuple(i)[0] for i in range(3)]
    orientation = np.array(iop).reshape(2, 3)
    directions = np.vstack((orientation, SliceVector)).T
    return _build_vtkTransform(oo, directions)


def build_vtkTransform_meta(meta):
    oo = meta['Origin']
    orientation = np.array(meta['ImageOrientationPatient']).reshape(2, 3)
    directions = np.vstack((orientation, meta['SliceVector'])).T
    return _build_vtkTransform(oo, directions)


def _build_vtkTransform(oo, directions):
    matrix4x4 = vtk.vtkMatrix4x4()
    matrix4x4.SetElement(0,0, directions[0,0])
    matrix4x4.SetElement(1,0, directions[1,0])
    matrix4x4.SetElement(2,0, directions[2,0])
    #
    matrix4x4.SetElement(0,1, directions[0,1])
    matrix4x4.SetElement(1,1, directions[1,1])
    matrix4x4.SetElement(2,1, directions[2,1])
    #
    matrix4x4.SetElement(0,2, directions[0,2])
    matrix4x4.SetElement(1,2, directions[1,2])
    matrix4x4.SetElement(2,2, directions[2,2])
    #
    matrix4x4.SetElement(0,3, oo[0])
    matrix4x4.SetElement(1,3, oo[1])
    matrix4x4.SetElement(2,3, oo[2])
    #
    matrix4x4.SetElement(3,0, 0.0)
    matrix4x4.SetElement(3,1, 0.0)
    matrix4x4.SetElement(3,2, 0.0)
    matrix4x4.SetElement(3,3, 1.0)
    #
    transFormMatrix = vtk.vtkTransform()
    transFormMatrix.SetMatrix(matrix4x4)
    return transFormMatrix


def vtiToVts_viaTransform(vtiObj, transMatrix):
    """
    Uses field data: ImageOrientationPatient, ImagePositionPatient
    :param vtiObj:
    :param transMatrix: vtk.vtkMatrix4x4: see build_vtkTransform_*
    :return:
    """
    vtiObj.SetOrigin(0.0,0.0,0.0) # Origin should be in the transMatrix
    ##
    tfilterMatrix = vtk.vtkTransformFilter()
    tfilterMatrix.SetTransform(transMatrix)
    tfilterMatrix.SetInputData(vtiObj)
    tfilterMatrix.Update()
    return tfilterMatrix.GetOutput()


def filterResampleToImage(vtsObj, target_spacing):
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


# ===================================================================================================
def __writerWrite(writer, data, fileName):
    writer.SetFileName(fileName)
    writer.SetInputData(data)
    writer.Write()
    return fileName


def writeNII(data, fileName):
    writer = vtk.vtkNIFTIImageWriter()
    return __writerWrite(writer, data, fileName)


def writeMHA(data, fileName):
    writer = vtk.vtkMetaImageWriter()
    return __writerWrite(writer, data, fileName)


def writeVTS(data, fileName):
    writer = vtk.vtkXMLStructuredGridWriter()
    return __writerWrite(writer, data, fileName)


def writeVTI(data, fileName):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetDataModeToBinary()
    return __writerWrite(writer, data, fileName)


def nii2vti(fullFileName):
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

def writeVtkFile(data, fileName):
    if fileName.endswith('.vti'):
        return writeVTI(data, fileName)
    elif fileName.endswith('.vts'):
        return writeVTS(data, fileName)
    elif fileName.endswith('.mha'):
        return writeMHA(data, fileName)
    
def readVTKFile(fileName):
    # --- CHECK EXTENSTION - READ FILE ---
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

def readImageStackToVTI(imageFileNames, meta, arrayName='PixelData', CONVERT_TO_GREYSCALE=False):
    append_filter = vtk.vtkImageAppend()
    append_filter.SetAppendAxis(2)  # Combine images along the Z axis
    for file_name in imageFileNames:
        thisImage = readVTKFile(file_name)
        append_filter.AddInputData(thisImage)
    append_filter.Update()
    combinedImage = append_filter.GetOutput()
    combinedImage.SetOrigin(meta.get('Origin', [0.0,0.0,0.0]))
    combinedImage.SetSpacing(meta.get('Spacing', meta.get('Resolution', [1.0,1.0,1.0])))
    a = getScalarsAsNumpy(combinedImage)
    if CONVERT_TO_GREYSCALE:
        a = np.mean(a, 1)
    addArrayFromNumpy(combinedImage, a, arrayName, SET_SCALAR=True)
    delArraysExcept(combinedImage, [arrayName])
    return combinedImage

# =========================================================================
##          PVD Stuff
# =========================================================================
def checkIfExtnPresent(fileName, extn):
    if (extn[0] == '.'):
        extn = extn[1:]
    le = len(extn)
    if (fileName[-le:] != extn):
        fileName = fileName + '.' + extn
    return fileName

def _writePVD(rootDirectory, filePrefix, outputSummary):
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


def _makePvdOutputDict(vtkDict, filePrefix, fileExtn, subDir=''):
    outputSummary = {}
    myKeys = vtkDict.keys()
    myKeys = sorted(myKeys)
    for timeId in range(len(myKeys)):
        fileName = __buildFileName(filePrefix, timeId, fileExtn)
        trueTime = myKeys[timeId]
        outputMeta = {'FileName': os.path.join(subDir, fileName), 'TimeID': timeId, 'TrueTime': trueTime}
        outputSummary[timeId] = outputMeta
    return outputSummary

def __writePvdData(vtkDict, rootDir, filePrefix, fileExtn, subDir=''):
    myKeys = vtkDict.keys()
    myKeys = sorted(myKeys)
    for timeId in range(len(myKeys)):
        fileName = __buildFileName(filePrefix, timeId, fileExtn)
        fileOut = os.path.join(rootDir, subDir, fileName)
        if type(vtkDict[myKeys[timeId]]) == str:
            os.rename(vtkDict[myKeys[timeId]], fileOut)
        else:
            writeVtkFile(vtkDict[myKeys[timeId]], fileOut)

def writeVtkPvdDict(vtkDict, rootDir, filePrefix, fileExtn, BUILD_SUBDIR=True):
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

def deleteFilesByPVD(pvdFile, FILE_ONLY=False, QUIET=False):
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

def __buildFileName(prefix, idNumber, extn):
    ids = '%05d'%(idNumber)
    if extn[0] != '.':
        extn = '.' + extn
    fileName = prefix + '_' + ids + extn
    return fileName

def readPVDFileName(fileIn, vtpTime=0.0, timeIDs=None, RETURN_OBJECTS_DICT=False):
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

def readPVD(fileIn, timeIDs=None):
    if timeIDs is None:
        timeIDs = []
    return readPVDFileName(fileIn, timeIDs=timeIDs, RETURN_OBJECTS_DICT=True)

def pvdGetDict(pvd, timeIDs=None):
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


def getPatientMatrixDictFromVTI(data, patMat):
    dx,dy,dz = data.GetSpacing()
    oo = data.GetOrigin()
    # 1st option from meta, then fielddata then default
    iop = patMat.get('ImageOrientationPatient', 
                        getFieldData(data, 
                                    'ImageOrientationPatient', 
                                    default=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    sliceVec = patMat.get('SliceVector', 
                        getFieldData(data, 
                                    'SliceVector', 
                                    default=[0.0, 0.0, 1.0]))
    patMat = {'PixelSpacing': [dx*1000.0, dy*1000.0],
                         'ImagePositionPatient': [i*1000.0 for i in oo],
                         'ImageOrientationPatient': iop,
                         'SpacingBetweenSlices': dz*1000.0,
                         'SliceVector': sliceVec}
    return patMat


# =========================================================================
# =========================================================================
## DICOM-SEG
# =========================================================================
def array_to_DcmSeg(arr, source_dicom_ds_list, dcmSegFileOut=None, algorithm_identification=None):
    if not HIGHDCM:
        raise ImportError("Missing highdicom\n Please run: pip install highdicom")
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
    meta = getDcmSeg_meta(ds)
    seg_data = ds.pixel_array
    seg_data = np.transpose(seg_data, axes=[2,1,0])
    image_data = vtk.vtkImageData()
    image_data.SetOrigin(meta["Origin"])
    image_data.SetDimensions(meta["Dimensions"])
    image_data.SetSpacing(meta["Spacing"])
    vtk_array = numpy_support.numpy_to_vtk(num_array=seg_data.flatten('F'), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    image_data.GetPointData().SetScalars(vtk_array)
    if TRUE_ORIENTATION:
        writeVTS(_vti2vts(image_data, meta), vtk_output_path)
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



## ---------------------------------------
def buildImage2PatientCoordinateMatrix(meta):
    dx, dy, dz = meta['Spacing']
    oo = meta['Origin']
    orientation = np.array(meta['ImageOrientationPatient'])
    iop = np.hstack((orientation, meta['SliceVector']))
    matrix = np.array([[iop[0]*dx, iop[3]*dy, iop[6]*dz, oo[0]], 
                        [iop[1]*dx, iop[4]*dy, iop[7]*dz, oo[1]], 
                        [iop[2]*dx, iop[5]*dy, iop[8]*dz, oo[2]], 
                        [0, 0, 0, 1]])
    return matrix
def _buildVTSGrid_mat(meta): # MATRIX CALC OF POINTS - ALTERNATIVE METHOD - NOT USED
    mat = buildImage2PatientCoordinateMatrix(meta)
    dims = meta['Dimensions']
    xv, yv, zv = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing='ij')
    coordinates = np.stack((xv, yv, zv), axis=-1)
    coordinates = coordinates.reshape(-1, 3, order='F')
    coordinates = np.hstack((coordinates, np.ones([coordinates.shape[0], 1])))
    npPts = coordinates @ mat.T
    vts_data = vtk.vtkStructuredGrid()
    vts_data.SetDimensions(dims[:3])
    vtk_points = vtk.vtkPoints()
    double_array = numpy_support.numpy_to_vtk(npPts[:, :3], 1)
    vtk_points.SetData(double_array)
    vts_data.SetPoints(vtk_points)
    return vts_data
## ---------------------------------------