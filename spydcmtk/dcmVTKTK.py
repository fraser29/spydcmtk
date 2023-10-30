"""
Created on MArch 2023 (rewrite from old module - remove reliance on VTKDICOM)

@author: fraser

Dicom to VTK conversion toolkit

"""

import os
import numpy as np
import pydicom as dicom
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

VTK_AVAILABLE = True
try:
    import vtk
    from vtk.util import numpy_support # type: ignore
except ImportError:
    VTK_AVAILABLE = False

import spydcmtk.dcmTools as dcmTools


# ===================================================================================================
# EXPOSED METHODS
# ===================================================================================================

def arrToVTI(arr, meta, ds=None, INCLUDE_MATRIX=True):
    """Convert array (+meta) to VTI dict (keys=times, values=VTI volumes). 

    Args:
        arr (np.array): Array of pixel data, shape: nR,nC,nSlice,nTime
        meta (dict): dictionary containing meta to be added as Field data
            meta = {'Spacing': list_3 -> resolution, 
                    'Origin': list_3 -> origin, 
                    'ImageOrientationPatient': list_6 -> ImageOrientationPatient, 
                    'Times': list_nTime -> times (can be missing if nTime=1)}
        ds (pydicom dataset [optional]): pydicom dataset to use to add dicom tags as field data
        INCLUDE_MATRIX (bool [True]) : Boolean to include transform matrix (from ImageOrientationPatient) 
                                        in the image data (as DirectionMatrix). 
    
    Returns:
        vtiDict

    Raises:
        ValueError: If VTK import not available
    """
    if not VTK_AVAILABLE:
        raise NoVtkError()
    dims = arr.shape
    vtkDict = {}
    timesUsed = []
    try:
        mat3x3 = _buildMatrix3x3(meta)
    except KeyError:
        # Silently catch error and write without DirectionMatrix
        INCLUDE_MATRIX = False
    for k1 in range(dims[-1]):
        newImg = vtk.vtkImageData()
        newImg.SetSpacing(meta['Spacing'][0] ,meta['Spacing'][1] ,meta['Spacing'][2])
        newImg.SetOrigin(meta['Origin'][0], meta['Origin'][1], meta['Origin'][2])
        if INCLUDE_MATRIX:
            newImg.SetDirectionMatrix(mat3x3)
        newImg.SetDimensions(dims[0] ,dims[1] ,dims[2])
        A3 = arr[:,:,:,k1]
        npArray = np.reshape(A3, np.prod(arr.shape[:3]), 'F').astype(np.int16)
        aArray = numpy_support.numpy_to_vtk(npArray, deep=1)
        aArray.SetName('PixelData')
        newImg.GetPointData().SetScalars(aArray)
        if ds is not None:
            addFieldDataFromDcmDataSet(newImg, ds)
        try:
            thisTime = meta['Times'][k1]
        except KeyError:
            thisTime = k1
        if thisTime in timesUsed:
            thisTime = k1
        timesUsed.append(thisTime)
        vtkDict[thisTime] = newImg
    return vtkDict


def writeArrToVTI(arr, meta, filePrefix, outputPath, ds=None, INCLUDE_MATRIX=True):
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
    vtkDict = arrToVTI(arr, meta, ds=ds, INCLUDE_MATRIX=INCLUDE_MATRIX)
    return writeVTIDict(vtkDict, outputPath, filePrefix)

def writeVTIDict(vtiDict, outputPath, filePrefix):
    times = sorted(vtiDict.keys())
    if len(times) > 1:
        return writeVtkPvdDict(vtiDict, outputPath, filePrefix, 'vti', BUILD_SUBDIR=True)
    else:
        fOut = os.path.join(outputPath, f'{filePrefix}.vti')
        return writeVTI(vtiDict[times[0]], fOut)
    

def __transformMfromFieldData(vtkObj):
    iop = [vtkObj.GetFieldData().GetArray('ImageOrientationPatient').GetTuple(i)[0] for i in range(6)]
    vecC = np.cross(iop[3:6], iop[:3])
    iop += vecC.tolist()
    ipp = vtkObj.GetOrigin()
    matrix = [iop[0], iop[3], iop[6], ipp[0],
              iop[1], iop[4], iop[7], ipp[1],
              iop[2], iop[5], iop[8], ipp[2],
              0,0,0,1]
    return matrix


def getTransFormMatrixFromFieldData(vtkObj):
    matrix = __transformMfromFieldData(vtkObj)
    transFormMatrix = vtk.vtkTransform()
    transFormMatrix.SetMatrix(matrix)
    return transFormMatrix


def getTransFormMatrixFromVTIObjDirectionMatrix(vtkObj):
    transFormMatrix = vtk.vtkTransform()
    oo = vtkObj.GetOrigin()
    matrix = [vtkObj.GetDirectionMatrix().GetElement(0,0), vtkObj.GetDirectionMatrix().GetElement(1,0), vtkObj.GetDirectionMatrix().GetElement(2,0), oo[0],
                vtkObj.GetDirectionMatrix().GetElement(0,1), vtkObj.GetDirectionMatrix().GetElement(1,1), vtkObj.GetDirectionMatrix().GetElement(2,1), oo[1],
                vtkObj.GetDirectionMatrix().GetElement(0,2), vtkObj.GetDirectionMatrix().GetElement(1,2), vtkObj.GetDirectionMatrix().GetElement(2,2), oo[2],
                0,0,0,1]
    transFormMatrix.SetMatrix(matrix)
    return transFormMatrix


def __matrixToDicomTagFieldData(vtiObj, matrix):
    iop = [matrix[0], matrix[4], matrix[8], matrix[1], 
           matrix[5], matrix[9], matrix[2], matrix[6], 
           matrix[10]] # TODO check this
    ipp = [matrix[3], matrix[7], matrix[11]]
    addFieldData(vtkObj=vtiObj, fieldVal=iop, fieldName='ImageOrientationPatient')
    addFieldData(vtkObj=vtiObj, fieldVal=ipp, fieldName='ImagePositionPatient')


def vtiToVts_viaTransform(vtiObj, transMatrix=None):
    """
    Uses field data: ImageOrientationPatient, ImagePositionPatient
    :param vtiObj:
    :param transMatrix: can pass or grab from field data
    :return:
    """
    if vtiObj.GetDirectionMatrix().IsIdentity():
        if transMatrix is None:
            transMatrix = getTransFormMatrixFromFieldData(vtiObj)
            # As we took IPP from field data, explicitlly set VTI origin to 0,0,0
            vtiObj.SetOrigin(0.0,0.0,0.0)
    else:
        transMatrix = getTransFormMatrixFromVTIObjDirectionMatrix(vtiObj)
        vtiObj.SetOrigin(0.0,0.0,0.0)
    ##
    tfilterMatrix = vtk.vtkTransformFilter()
    tfilterMatrix.SetTransform(transMatrix)
    tfilterMatrix.SetInputData(vtiObj)
    tfilterMatrix.Update()
    return tfilterMatrix.GetOutput()

def _buildMatrix3x3(meta):
    iop = np.zeros((3,3))
    iop[:,0] = meta['ImageOrientationPatient'][0:3]
    iop[:,1] = meta['ImageOrientationPatient'][3:6]
    iop[:,2] = np.cross(iop[:,1], iop[:,0])
    mat = vtk.vtkMatrix3x3()
    for i in range(3):
        for j in range(3):
            mat.SetElement(i, j, iop[i,j])
    return mat

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


def addArrayFromNumpy(data, npArray, arrayName):
    aArray = numpy_support.numpy_to_vtk(npArray, deep=1)
    aArray.SetName(arrayName)
    data.GetPointData().AddArray(aArray)


def addFieldData(vtkObj, fieldVal, fieldName):
    tagArray = numpy_support.numpy_to_vtk(np.array([float(fieldVal)]))
    tagArray.SetName(fieldName)
    vtkObj.GetFieldData().AddArray(tagArray)


def getFieldData(vtkObj, fieldName):
    return numpy_support.vtk_to_numpy(vtkObj.GetFieldData().GetArray(fieldName)).copy()


def addFieldDataFromDcmDataSet(vtkObj, ds):
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


def getPatientMatrixDict(data, patMat):
    dx,dy,dz = data.GetSpacing()
    oo = data.GetOrigin()
    try: # Try from passed Matrix first , else field data, else axis aligned
        iop = getFieldData(data, 'ImageOrientationPatient')
    except AttributeError:
        iop = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    iop = patMat.get('ImageOrientationPatient', iop)
    patMat = {'PixelSpacing': [dx*1000.0, dy*1000.0],
                         'ImagePositionPatient': [i*1000.0 for i in oo],
                         'ImageOrientationPatient': iop,
                         'SpacingBetweenSlices': dz*1000.0}
    return patMat


def testVTK():
    if not VTK_AVAILABLE:
        raise NoVtkError()

class NoVtkError(Exception):
    ''' NoVtkError
            If VTK import fails '''
    def __init__(self):
        pass
    def __str__(self):
        return 'NoVtkError: VTK not found. Run: "pip install vtk"'
