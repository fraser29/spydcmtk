"""
Created on 4 Aug 2019

@author: fraser

Dicom to VTK conversion toolkit
THIS NEEDS VTK BUILT WITH VTK-DICOM

"""

import os
import vtk
from vtk.util import numpy_support
import numpy as np
import argparse
import datetime  ## FOR DEBUG
import base64
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET



# =========================================================================
def getRootDirWithSEdirs(startDir):
    """
    Search from startDir until find rootDir with format of subdirs:
        SE123_
        ... etc
    param1: start directory of search
    return: rootdirectory with subfolders of SE{int}_ format (startDir if not found)
    """

    def __isSEDirFormat(dd):
        if dd[:2] == "SE":
            try:
                int(dd.split("_")[0][2:])
            except ValueError:
                return False
            return True
        return False

    dicomRootDir = startDir
    for root, dirs, _ in os.walk(startDir):
        if any([__isSEDirFormat(dd) for dd in dirs]):
            dicomRootDir = root
            break
    return dicomRootDir


def seriesNumbersToDicomDirList(dicomRootDir, seriesNumbers):
    if not type(seriesNumbers) == list:
        seriesNumbers = [seriesNumbers]
    dicomRootDir = getRootDirWithSEdirs(dicomRootDir)
    SEList = os.listdir(dicomRootDir)
    dicomDirs = []
    for iSE in seriesNumbers:
        ii = [jj for jj in SEList if "SE%d" % (iSE) in jj.split('_')]
        dicomDirs.append(os.path.join(dicomRootDir, ii[0]))
    return dicomDirs


def __writerWrite(writer, data, fileName):
    writer.SetFileName(fileName)
    writer.SetInputData(data)
    writer.Write()
    return fileName


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
    
    if fileName.endswith('.vts'):
        return writeVTS(data, fileName)
    
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
## CONSTANTS
# =========================================================================
class DicomTags(object):
    # these are based on keyword value
    Modality = 0x0008, 0x0060
    Manufacturer = 0x0008, 0x0070
    ManufacturerModelName = 0x0008, 0x1090
    SoftwareVersions = 0x0018, 0x1020
    StudyDescription = 0x0008, 0x1030
    SeriesDescription = 0x0008, 0x103e
    BodyPartExamined = 0x0018, 0x0015
    SliceThickness = 0x0018, 0x0050
    RepetitionTime = 0x0018, 0x0080
    EchoTime = 0x0018, 0x0081
    NumberOfAverages = 0x0018, 0x0083
    MagneticFieldStrength = 0x0018, 0x0087
    SpacingBetweenSlices = 0x0018, 0x0088
    TriggerTime = 0x0018, 0x1060
    NominalInterval = 0x0018, 0x1062
    HeartRate = 0x0018, 0x1088
    CardiacNumberOfImages = 0x0018, 0x1090
    TriggerWindow = 0x0018, 0x1094
    ReceiveCoilName = 0x0018, 0x1250
    AcquisitionMatrix = 0x0018, 0x1310
    FlipAngle = 0x0018, 0x1314
    PatientPosition = 0x0018, 0x5100
    ImagePositionPatient = 0x0020, 0x0032
    ImageOrientationPatient = 0x0020, 0x0037
    StudyInstanceUID = 0x0020, 0x000d
    SeriesInstanceUID = 0x0020, 0x000e
    SeriesNumber = 0x0020, 0x0011
    PixelSpacing = 0x0028, 0x0030
    StudyDate = 0x0008, 0x0020
    PatientName = 0x0010, 0x0010
    PatientID = 0x0010, 0x0020
    PatientDateOfBirth = 0x0010, 0x0030
    PatientSex = 0x0010, 0x0040


def getTagCode(tagName):
    return eval("DicomTags.%s" % (tagName))


def getStdDicomTags():
    allVar = vars(DicomTags)
    res = []
    for iVar in allVar:
        val = getTagCode(iVar)
        if type(val) == tuple:
            if len(val) == 2:
                res.append(iVar)
            # print(iVar, val)
    return res


def getDicomTagsDict():
    tt = getStdDicomTags()
    return dict(zip([i for i in tt], [eval("DicomTags.%s" % (i)) for i in tt]))


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

# =========================================================================
## DICOM CONVERSION
# =========================================================================
def _dirToVtkFileNameArray(DIR, mustHaveTag=None):
    filenames1 = os.listdir(DIR)
    filenames1 = [os.path.join(DIR, i) for i in filenames1]
    filenamesU = []
    if mustHaveTag is not None:
        for iFile in filenames1:
            fr = getDicomReaderFromFileName(iFile)
            if getTag(fr, mustHaveTag[0], mustHaveTag[1]).IsValid():
                filenamesU.append(iFile)
    else:
        filenamesU = filenames1
    filenArray1 = vtk.vtkStringArray()
    filenArray1.SetNumberOfValues(len(filenamesU))
    for k1, iFile in enumerate(filenamesU):
        filenArray1.SetValue(k1, iFile)
    return filenArray1


def getDicomReaderFromDirectory(DIR, mustHaveTag=None):
    filenArray = _dirToVtkFileNameArray(DIR, mustHaveTag)
    return getDicomReaderFromFileArray(filenArray)


def getDicomReaderFromFileName(fileName):
    filenArray1 = vtk.vtkStringArray()
    filenArray1.SetNumberOfValues(1)
    filenArray1.SetValue(0, fileName)
    return getDicomReaderFromFileArray(filenArray1)


def getDicomReaderFromFileArray(fileNameArray):
    """
    :param fileNameArray: vtk.vtkStringArray
    :return: vtk.vtkDICOMReader
    """
    ss = vtk.vtkDICOMFileSorter()
    ss.SetInputFileNames(fileNameArray)
    ss.Update()
    i = ss.GetNumberOfSeries()
    if (i > 0):
        sortedFiles1 = ss.GetFileNamesForSeries(0)
        # print('N files', sortedFiles1.GetNumberOfValues())
        readerMag = vtk.vtkDICOMReader()
        readerMag.SetFileNames(sortedFiles1)
        readerMag.Update()
        return readerMag
    else:
        raise IOError


def isPCImage(dicomReader):
    if IS_GE(dicomReader):
        return getTag(dicomReader, 0x0043, 0x1030).AsInt() > 8  # FIXME (not totally sure this is correct)
        # return getMinPossiblePixel(dicomReader) < -5
    elif IS_BRUKER(dicomReader):
        return (getTag(dicomReader, 0x0020, 0x0013).AsInt() % 2) == 1


def getPCMagAndPhaseFileArraysFromDir(DIR):
    """
    Will split dicom files in DIR into two vtkStringArrays - mag and then phase
    Decide based on lowest pixel val (not sure of another way...)
     - this will die with match report if not single series phase/magnitude...
    :param DIR:
    :return: list [vtk.vtkStringarray]
    """
    filenames1 = os.listdir(DIR)
    filenArray1 = vtk.vtkStringArray()
    filenArray2 = vtk.vtkStringArray()
    filenArray1.SetNumberOfValues(int(len(filenames1) / 2))
    filenArray2.SetNumberOfValues(int(len(filenames1) / 2))
    c1, c2 = 0, 0
    for k1, iFile in enumerate(filenames1):
        ff = os.path.join(DIR, iFile)
        rr = getDicomReaderFromFileName(ff)
        # print(iFile, getTag(rr, 0x0043,0x1030).AsInt())
        if isPCImage(rr):
            filenArray2.SetValue(c2, ff)
            c2 += 1
        else:
            filenArray1.SetValue(c1, ff)
            c1 += 1
    return [filenArray1, filenArray2]


def getTag(dicomReader, tagInt1, tagInt2=None, RETURN_LIST=False):
    """
    Get a dicom tag from the supplied reader.
    :param dicomReader: dicomreader
    :param tagInt1: tag code: e.g 0x0008  -- this can also be tuple of both codes - for ease of use)
    :param tagInt2: tag code number two (only if tag one is not tuple/list)
    :param RETURN_LIST: will return tag value from every instance in reader
    :return: a vtk.vtkDICOMValue   - can convert to float/str etc using eg .AsInt(). Can check with method: .IsValid()
    """
    try:
        tagInt2 = tagInt1[1]
        tagInt1 = tagInt1[0]
    except TypeError:
        pass
    meta = dicomReader.GetMetaData()
    fileMap = dicomReader.GetFileIndexArray()
    fileIndex = fileMap.GetComponent(0, 0)
    frameMap = dicomReader.GetFrameIndexArray()
    frameIndex = int(frameMap.GetComponent(0, 0))
    tagID = vtk.vtkDICOMTag(tagInt1, tagInt2)
    if RETURN_LIST:
        return [meta.GetAttributeValue(i, frameIndex, tagID) for i in range(fileMap.GetNumberOfValues())]
    else:
        tagValue = meta.GetAttributeValue(int(fileIndex), frameIndex, tagID)
        return tagValue


def getVendor(dicomReader):
    return getTag(dicomReader, 0x0008, 0x0070).AsString()


def IS_SIEMENS(dicomReader):
    if 'siem' in getVendor(dicomReader).lower():
        return True
    return False


def IS_GE(dicomReader):
    if 'ge' in getVendor(dicomReader).lower():
        return True
    return False


def IS_PHILIPS(dicomReader):
    if 'phil' in getVendor(dicomReader).lower():
        return True
    return False


def IS_BRUKER(dicomReader):
    if 'bruk' in getVendor(dicomReader).lower():
        return True
    return False


def getVenc(dicomReader):
    try:
        if IS_GE(dicomReader):
            TagGE = getTag(dicomReader, 0x0019, 0x10cc).AsInt()
            try:
                v_mmps = TagGE
            except ValueError:
                v_mmps = int("".join(reversed(TagGE)).encode('hex'), 16)
            return v_mmps / 1000.0
        elif IS_PHILIPS(dicomReader):
            TagPHIL = getTag(dicomReader, 0x2001, 0x101a).AsDouble()
            for iV in TagPHIL:
                if int(iV) > 0:
                    return int(iV) / 100.0
        elif IS_SIEMENS(dicomReader):  # NOTE  - must be reader from one of phase series
            TagSIEM = getTag(dicomReader, 0x0018, 0x0024).AsString()
            usv = TagSIEM.find('_v')
            return float(TagSIEM[usv + 2:-2]) / 100.0
        elif IS_BRUKER(dicomReader):
            TagBRUK = getTag(dicomReader, 0x0028, 0x1050).AsString()
            return float(TagBRUK)
    except (KeyError, ValueError):
        return None


def getMaxPossiblePixel(dicomReader):
    if IS_SIEMENS(dicomReader):
        return 2 ** getTag(dicomReader, 0x0028, 0x0101).AsInt()
    elif IS_GE(dicomReader):
        return getVenc(dicomReader) * 1000.0
    elif IS_BRUKER(dicomReader):
        return 2 ** getTag(dicomReader, 0x0028, 0x0107).AsInt()
    else:
        print('RETURNING DEFAULT MAXPOSSIBLEPIXEL (2DPC) - VENDOR IS %s' % (getVendor(dicomReader)))
        return 4096


def getMinPossiblePixel(dicomReader):
    return getTag(dicomReader, 0x0028, 0x0106).AsInt()


def getImageOrientationPatient(dicomReader):
    return [float(i) for i in getTag(dicomReader, 0x0020, 0x0037).AsString().split('\\')]


def getImagePositionPatient(dicomReader):
    return [float(i) for i in getTag(dicomReader, 0x0020, 0x0032).AsString().split('\\')]


def getImageCenter(dicomReader):
    # Will returrn in real world CS meters
    vtsDict = cineReaderToVTK(dicomReader, None)
    return list(vtsDict.values())[0].GetCenter()


def getSliceLocations(dicomReader):
    sliceList = []
    for k1 in range(dicomReader.GetFileIndexArray().GetNumberOfTuples()):
        sliceList.append(dicomReader.GetMetaData().GetAttributeValue(k1, 0, vtk.vtkDICOMTag(0x0020, 0x1041)).AsFloat())
    return sliceList


def getNumberOfSlices(dicomReader):
    """
    Uses slice location
    :param dicomReader:
    :return: int
    """
    return len(set(getSliceLocations(dicomReader)))


def getDimensions(dicomReader):
    """
    :param dicomReader:
    :return: nRows, nCols, nSlices, nTimeSteps
    """
    return (getTag(dicomReader, 0x0028, 0x0010).AsInt(),
            getTag(dicomReader, 0x0028, 0x0011).AsInt(),
            getNumberOfSlices(dicomReader),
            dicomReader.GetTimeDimension())


def getResolution(dicomReader):
    return getPixelSpacing(dicomReader) + [getTag(dicomReader, 0x0018, 0x0088).AsFloat()]


def getPixelSpacing(dicomReader):
    return [float(i) for i in getTag(dicomReader, 0x0028, 0x0030).AsString().split('\\')]


def getImageNormal(dicomReader):
    v1v2 = getImageOrientationPatient(dicomReader)
    return np.cross(v1v2[:3], v1v2[3:])


def getVelocityEncodeScale(dicomReader):
    return getTag(dicomReader, 0x0019, 0x10e2).AsFloat()


def getVASFlags(dicomReader):
    return getTag(dicomReader, 0x0043, 0x1032).AsFloat()


def getVelocityConversionSlope(dicomReader):
    if IS_GE(dicomReader):
        return 0.001
    elif IS_SIEMENS(dicomReader):
        return (2.0 * getVenc(dicomReader)) / (2.0 * getMaxPossiblePixel(dicomReader))
    elif IS_PHILIPS(dicomReader):
        return 0.0
    elif IS_BRUKER(dicomReader):
        return 0.001
    else:
        return (2.0 * getVenc(dicomReader)) / (getMaxPossiblePixel(dicomReader) - getMinPossiblePixel(dicomReader))


def getVelocityConversionOffset(dicomReader):
    # FIXME
    if IS_GE(dicomReader):
        return 0.0
    elif IS_SIEMENS(dicomReader):
        return 0.0
    elif IS_PHILIPS(dicomReader):
        return 0.0
    elif IS_BRUKER(dicomReader):
        return 0.0
    else:
        # return -getVenc(dicomReader)-(getVelocityConversionSlope(dicomReader)*getMinPossiblePixel(dicomReader))
        return 0.0


def __getDicomTo4DFlowVendorSpecificModifiers(dicomReader):
    modifiers = [1, 1, 1]
    if IS_SIEMENS(dicomReader):
        return [1, 1, 1]
    elif IS_PHILIPS(dicomReader):
        return [1, 1, 1]
    elif IS_GE(dicomReader):
        return [1, 1, -1]
    return modifiers


def getHeartRateFromSlice_bpm(dicomReader):
    try:
        if IS_SIEMENS(dicomReader):
            return 60.0 / (getTag(dicomReader, 0x0018, 0x1062).AsDouble() / 1000.0)
        elif IS_PHILIPS(dicomReader):
            return getTag(dicomReader, 0x0018, 0x1088).AsDouble()
        elif IS_GE(dicomReader):
            return getTag(dicomReader, 0x0018, 0x1088).AsDouble()
    except ZeroDivisionError:
        return 0.0
    except KeyError:
        return 0.0
    raise ValueError


def getTriggerTime(dicomReader):
    return getTag(dicomReader, 0x0018, 0x1060).AsDouble()


def getSeriesTime(dicomReader):
    ii = getTag(dicomReader, 0x0008, 0x0031).AsString()
    # return datetime.datetime.strptime(ii, '%H%M%S')
    return ii


def getTagValuesDictForReader(dicomReader, tagsDict=None):
    """
    :param dicomReader: dicomReader
    :param tagsDict: dict of tagName, tagCode [if none use default]
    :return: dict of tagName, value
    """
    if tagsDict is None:
        tagsDict = getDicomTagsDict()
    dictOut = {}
    for iTag in tagsDict.keys():
        val = getTag(dicomReader, tagsDict[iTag])
        if val.IsValid():
            dictOut[iTag] = val.AsString()
        else:
            print(iTag)
    venc = getVenc(dicomReader)
    if venc is not None:
        dictOut["VENC_mps"] = venc
    return dictOut


def getPatientMatrix(dicomDir):
    """Read files in directory, build vtkMatrix4x4

    Args:
        dicomDir (str): full path to directory of dicom series

    Returns:
        vtkMatrix4x4: matrix from image-orientation-patient and image-position-patient
    """
    reader = getDicomReaderFromDirectory(dicomDir)
    return reader.GetPatientMatrix()


def _addFieldData(vtkObj, dicomReader):
    tagsDict = getDicomTagsDict()
    for iTag in tagsDict.keys():
        val = getTag(dicomReader, tagsDict[iTag])
        if val.IsValid():
            try:
                valSList = [float(i) for i in val.AsString().split("\\")]
                tagArray = numpy_support.numpy_to_vtk(np.array(valSList))
            except (TypeError, ValueError):
                tagArray = vtk.vtkStringArray()
                tagArray.SetNumberOfValues(1)
                tagArray.SetValue(0, val.AsString())
            tagArray.SetName(iTag)
            vtkObj.GetFieldData().AddArray(tagArray)
    ## VENC
    try:
        venc = getVenc(dicomReader)
        tagArray = numpy_support.numpy_to_vtk(np.array([float(venc)]))
        tagArray.SetName("VENC")
    except TypeError:
        pass
    vtkObj.GetFieldData().AddArray(tagArray)


def printTableOfMainTags(dicomDir):
    reader = getDicomReaderFromDirectory(dicomDir)
    tagsDict = getTagValuesDictForReader(reader)
    for iK in sorted(tagsDict.keys()):
        print('%50s, %50s' % (iK, tagsDict[iK]))


def flipImageDataAlongAxis(img, axID):
    flipper = vtk.vtkImageFlip()
    flipper.SetFilteredAxes(axID)
    flipper.SetInputData(img)
    flipper.Update()
    return flipper.GetOutput()


# =========================================================================
# =========================================================================
def cineReaderToVTK(readerCINE, scratchDir=None, RETURN_VTI=False, DEBUG=False, timeIDs=[]):
    numberOfTimeSlots = readerCINE.GetTimeDimension()
    print('Building CINE VTK dict with %d time steps' % (numberOfTimeSlots))
    dictOut = {}
    if len(timeIDs) == 0:
        timeIDs = range(numberOfTimeSlots)
    else:
        for k1 in range(len(timeIDs)):
            if timeIDs[k1] < 0:
                timeIDs[k1] = numberOfTimeSlots + timeIDs[k1]
    for k1 in timeIDs:
        if DEBUG:
            print('  Time step: %d' % (k1))
        readerCINE.SetDesiredTimeIndex(k1)
        readerCINE.Update()
        thisTime = getTriggerTime(readerCINE) / 1000.0
        if RETURN_VTI:
            # readerCINE.SetMemoryRowOrderToFileNative()
            # readerCINE.Update()
            vvi = readerCINE.GetOutput()
            vo = vtk.vtkImageData()
            vo.DeepCopy(vvi)
            # vo = flipImageDataAlongAxis(vo, 1)
            _addFieldData(vo, readerCINE)
            if scratchDir is not None:
                fOut = writeVTI(vo, os.path.join(scratchDir, "data%d.vti" % (k1)))
                dictOut[thisTime] = fOut
            else:
                dictOut[thisTime] = vo
        else:
            # APPLY TRANSFORM
            readerCINE.SetMemoryRowOrderToFileNative()
            readerCINE.Update()
            vvi = readerCINE.GetOutput()
            matrix = readerCINE.GetPatientMatrix()
            ##====
            transMatrix = vtk.vtkTransform()
            transMatrix.SetMatrix(matrix)
            tfilterMatrix = vtk.vtkTransformFilter()
            tfilterMatrix.SetTransform(transMatrix)
            tfilterMatrix.SetInputData(vvi)
            tfilterMatrix.Update()
            ##
            transScale = vtk.vtkTransform()
            transScale.Scale(0.001, 0.001, 0.001)
            tfilterScale = vtk.vtkTransformFilter()
            tfilterScale.SetTransform(transScale)
            tfilterScale.SetInputData(tfilterMatrix.GetOutput())
            tfilterScale.Update()
            vvs = tfilterScale.GetOutput()
            ##====
            _addFieldData(vvs, readerCINE)
            ##====
            if scratchDir is not None:
                fOut = writeVTS(vvs, os.path.join(scratchDir, "data%d.vts" % (k1)))
                dictOut[thisTime] = fOut
            else:
                dictOut[thisTime] = vvs
    return dictOut


def _pcReadersToVTK(readerMag, readerPh, THICKNESS=None, QUIET=False):
    numberOfTimeSlots = readerMag.GetTimeDimension()
    if not QUIET:
        print('Building 2DPC VTK dict with %d time steps' % (numberOfTimeSlots))
    dictOut = {}
    timeIDsToWrite = range(numberOfTimeSlots)
    for k1 in timeIDsToWrite:
        readerMag.SetDesiredTimeIndex(k1)
        readerMag.Update()
        thisTime = getTriggerTime(readerMag) / 1000.0
        ##
        readerPh.SetDesiredTimeIndex(k1)
        readerPh.Update()
        #
        VCO = getVelocityConversionOffset(readerPh)
        VCS = getVelocityConversionSlope(readerPh)
        # print(thisTime, 'vOffset',VCO, 'vSlope',VCS)
        ##
        # APPLY TRANSFORM
        vvs = __transformDicomReaderToVTS(readerMag)
        ##==
        velVts = __transformDicomReaderToVTS(readerPh)
        pixData = getArrayAsNumpy(velVts, 'PixelData')
        if False:  # getVASFlags(readerPh) > 1.5:
            vScale = np.pi * getVelocityEncodeScale(readerPh) / (getVenc(readerPh))
            mPix = getArrayAsNumpy(vvs, 'PixelData')
            # iVels2 = (pixData / mPix) / vScale
            print("USING GE MAG_SCALING & vSCALE = ", vScale)
            iVels = (pixData / mPix) / vScale
        else:
            iVels = pixData * VCS + VCO
        iop = getImageOrientationPatient(readerPh)
        vecN = np.cross(iop[3:], iop[:3])
        vvnp = np.array([i * vecN for i in iVels])
        vecArray = numpy_support.numpy_to_vtk(vvnp, deep=1)
        vecArray.SetName('Vel')
        vvs.GetPointData().SetVectors(vecArray)
        # vvnp2 = np.array([i * vecN for i in iVels2])
        phaseArray = numpy_support.numpy_to_vtk(pixData, deep=1)
        phaseArray.SetName('Phase')
        vvs.GetPointData().AddArray(phaseArray)
        ##====
        _addFieldData(vvs, readerMag)
        ##====
        dictOut[thisTime] = vvs
    return dictOut


def __transformDicomReaderToVTS(reader):
    """
    Take a reader and convert to VTS object
    :param reader: vtkDICOMReader
    :return: vts object in scanner (real world) coordinate system
    """
    reader.SetMemoryRowOrderToFileNative()
    reader.Update()
    matrix = reader.GetPatientMatrix()
    # ===
    transMatrix = vtk.vtkTransform()
    transMatrix.SetMatrix(matrix)
    tfilterMatrix = vtk.vtkTransformFilter()
    tfilterMatrix.SetTransform(transMatrix)
    tfilterMatrix.SetInputData(reader.GetOutput())
    tfilterMatrix.Update()
    ##
    transScale = vtk.vtkTransform()
    transScale.Identity()
    transScale.Scale(0.001, 0.001, 0.001)
    tfilterScale = vtk.vtkTransformFilter()
    tfilterScale.SetTransform(transScale)
    tfilterScale.SetInputData(tfilterMatrix.GetOutput())
    tfilterScale.Update()
    return tfilterScale.GetOutput()


def _fourDFlowReadersToVTK(readerMag, dicomReader_RL, dicomReader_AP, dicomReader_FH,
                            scratchDir, QUIET=True, DEBUG=False):
    """

    :param readerMag: vtkDICOMReader of magnitude data
    :param dicomReader_RL: vtkDICOMReader of right-left data
    :param dicomReader_AP: vtkDICOMReader of anterior-posterior data
    :param dicomReader_FH: vtkDICOMReader of feet-head data
    :param RETURN_VTI: NOT IN USE
    :return: dictionary of k,v = times,vtsObject with 'PixelData' scalar and 'Vel' vector
    """
    numberOfTimeSlots = readerMag.GetTimeDimension()
    if not QUIET:
        print('Building fourDFlow VTK dict with %d time steps' % (numberOfTimeSlots))
    dictOut = {}
    timeIDsToWrite = range(numberOfTimeSlots)
    if DEBUG: timeIDsToWrite = [2]
    for k1 in timeIDsToWrite:
        readerMag.SetDesiredTimeIndex(k1)
        readerMag.Update()
        thisTime = getTriggerTime(readerMag) / 1000.0
        ##
        dicomReader_RL.SetDesiredTimeIndex(k1)
        dicomReader_RL.Update()
        dicomReader_AP.SetDesiredTimeIndex(k1)
        dicomReader_AP.Update()
        dicomReader_FH.SetDesiredTimeIndex(k1)
        dicomReader_FH.Update()
        #
        VCO = getVelocityConversionOffset(dicomReader_RL)
        VCS = getVelocityConversionSlope(dicomReader_RL)
        modifiers = __getDicomTo4DFlowVendorSpecificModifiers(dicomReader_RL)
        if DEBUG:
            print(getTag(dicomReader_RL, 0x0018, 0x0024).AsString(), getMaxPossiblePixel(dicomReader_AP),
                    getMinPossiblePixel(dicomReader_AP))
            print(getVendor(dicomReader_AP), getVenc(dicomReader_AP), VCO, VCS, modifiers)
        ##
        # APPLY TRANSFORM
        vvs = __transformDicomReaderToVTS(readerMag)
        ##==
        vecs = []
        for iReader in [dicomReader_RL, dicomReader_AP, dicomReader_FH]:
            velVts = __transformDicomReaderToVTS(iReader)
            pixData = getArrayAsNumpy(velVts, 'PixelData')
            iVels = pixData * VCS + VCO
            if DEBUG:
                print('    ', getTag(iReader, 0x0018, 0x0024).AsString(), getMaxPossiblePixel(dicomReader_AP),
                        getMinPossiblePixel(dicomReader_AP))
                print('    ', getVendor(iReader), getVenc(dicomReader_AP), VCO, VCS, min(pixData), max(pixData),
                            min(iVels), max(iVels))
            vecs.append(iVels)
        vvnp = np.vstack((modifiers[0] * vecs[0], modifiers[1] * vecs[1], modifiers[2] * vecs[2])).T.copy()
        vecArray = numpy_support.numpy_to_vtk(vvnp, deep=1)
        vecArray.SetName('Vel')
        vvs.GetPointData().SetVectors(vecArray)
        ##====
        _addFieldData(vvs, readerMag)
        ##====
        fOut = writeVTS(vvs, os.path.join(scratchDir, "data%d.vts" % (k1)))
        dictOut[thisTime] = fOut
    return dictOut


def getArrayAsNumpy(data, arrayName):
    return numpy_support.vtk_to_numpy(data.GetPointData().GetArray(arrayName)).copy()


def addArrayFromNumpy(data, npArray, arrayName):
    aArray = numpy_support.numpy_to_vtk(npArray, deep=1)
    aArray.SetName(arrayName)
    data.GetPointData().AddArray(aArray)


def dicomDirToVTKImageData(dicomDirectory, timeIndex=0):
    """Note is not in correct coordinate system, but has field data to describe

    Args:
        dicomDirectory ([type]): [description]
    """
    dcmReader = getDicomReaderFromDirectory(dicomDirectory)
    dcmReader.SetDesiredTimeIndex(timeIndex)
    dcmReader.Update()
    ii = dcmReader.GetOutput()
    _addFieldData(ii, dcmReader)
    return ii


def dicomDirTo3DNumPy(dicomDirectory):
    ii = dicomDirToVTKImageData(dicomDirectory)
    A = getArrayAsNumpy(ii, ii.GetPointData().GetScalars().GetName())
    return np.reshape(A, ii.GetDimensions(), 'F')


def dicomDirToCinePVD(cineDicomDir, outputDir=None, outputName=None, RETURN_VTI=False, mustHaveTag=None, timeIDs=[]):
    rr = getDicomReaderFromDirectory(cineDicomDir, mustHaveTag)
    vtkDict = cineReaderToVTK(rr, outputDir, RETURN_VTI, timeIDs=timeIDs)
    if outputDir is not None:
        formatExt = "vts"
        if RETURN_VTI:
            formatExt = "vti"
        if len(vtkDict) > 1:
            return writeVtkPvdDict(vtkDict, outputDir, outputName, formatExt, True)
        else:
            tempFile = list(vtkDict.values())[0]
            outFile = os.path.join(outputDir, outputName + '.' + formatExt)
            os.rename(tempFile, outFile)
            return outFile
    else:
        return vtkDict


def dicomDirTo2DPCPVD(pcDicomDirs, outputDir=None, outputName=None):
    """

    :param pcDicomDirs: list of dirs ** if not list then just string to dir and will split to phase mag automatically **
    :param outputDir:
    :param outputName:
    :return: fileName if given to write else vtkDict
    """
    if type(pcDicomDirs) != list:
        pcDicomFilesList = getPCMagAndPhaseFileArraysFromDir(pcDicomDirs)
    else:
        pcDicomFilesList = [_dirToVtkFileNameArray(iDir) for iDir in pcDicomDirs]
    rrM, rrP = getDicomReaderFromFileArray(pcDicomFilesList[0]), getDicomReaderFromFileArray(pcDicomFilesList[1])
    vtkDict = _pcReadersToVTK(rrM, rrP)
    if outputDir:
        formatExt = "vts"
        return writeVtkPvdDict(vtkDict, outputDir, outputName, formatExt, True)
    else:
        return vtkDict


def dicomDirToFDFPVD(dicomDirList, outputDir, outputName, QUIET=False, DEBUG=False):
    """
    Convert list of dicom directories to 4D flow PVD
    :param dicomDirList: list-4 folder names - [Mag, RL, AP, FH]
    :param outputDir:
    :param outputName:
    :param QUIET:
    :param DEBUG:
    :return:
    """
    if not QUIET:
        print('Begin read: %s' % (datetime.datetime.strftime(datetime.datetime.now(), '%H:%M:%S')))
    rM = getDicomReaderFromDirectory(dicomDirList[0])
    rRL = getDicomReaderFromDirectory(dicomDirList[1])
    rAP = getDicomReaderFromDirectory(dicomDirList[2])
    rFH = getDicomReaderFromDirectory(dicomDirList[3])
    if not QUIET:
        print('Begin recon: %s' % (datetime.datetime.strftime(datetime.datetime.now(), '%H:%M:%S')))
    vtkDict = _fourDFlowReadersToVTK(rM, rRL, rAP, rFH, outputDir, QUIET, DEBUG)
    if not QUIET:
        print('Begin write: %s' % (datetime.datetime.strftime(datetime.datetime.now(), '%H:%M:%S')))
    extn = 'vts'
    fOut = writeVtkPvdDict(vtkDict, outputDir, outputName, extn, True)
    if not QUIET:
        print('END: %s' % (datetime.datetime.strftime(datetime.datetime.now(), '%H:%M:%S')))
    return fOut


def getPatientMatrixFromFieldData(imageData):
    def __fieldDataToNP(data, arrayName):
        return numpy_support.vtk_to_numpy(data.GetFieldData().GetArray(arrayName))

    ipp = __fieldDataToNP(imageData, 'ImagePositionPatient')
    iop = __fieldDataToNP(imageData, 'ImageOrientationPatient')
    nn = np.cross(iop[:3], iop[3:])
    # nn = np.cross(iop[3:],iop[:3])

    mat = vtk.vtkMatrix4x4()
    mat.DeepCopy((iop[0], iop[3], nn[0], ipp[0],
                iop[1], iop[4], nn[1], ipp[1],
                iop[2], iop[5], nn[2], ipp[2],
                0, 0, 0, 1))
    return mat


def writeImageDataToDicoms(dicomTemplateFile, imageData, outputDirectory, arrayName, newSeNum,
                            deltaT=1, TEMPORAL=False):
    """
    Have tested for reader to VTI. WORKS. 16/04/2020
    :param dicomTemplateFile:
    :param imageData:
    :param outputDirectory:
    :param arrayName:
    :param newSeNum: can use getTag(reader, DicomTags.SeriesNumber).AsInt()
    :param deltaT: time delta [default 1]
    :param TEMPORAL: Will write vector data as times
    :return:
    """

    mrG = vtk.vtkDICOMMRGenerator()
    patientMatrix = None
    try:
        reader = getDicomReaderFromFileName(dicomTemplateFile)
    except IOError:
        reader = getDicomReaderFromDirectory(dicomTemplateFile)
        patientMatrix = reader.GetPatientMatrix()
    meta = reader.GetMetaData()
    ## Adjust the series number
    newSeNumStr = '%05d' % (newSeNum)
    i0, i1 = DicomTags.SeriesNumber
    tag = vtk.vtkDICOMTag(int(i0), int(i1))
    meta.SetAttributeValue(tag, newSeNumStr)
    # if triggerTime is not None:
    #     i0, i1 = DicomTags.TriggerTime
    #     tag = vtk.vtkDICOMTag(int(i0), int(i1))
    #     meta.SetAttributeValue(tag, triggerTime)
    if patientMatrix is None:
        patientMatrix = getPatientMatrixFromFieldData(imageData)

    ## Reduce to only array of interest and set as int16
    for ia in [imageData.GetPointData().GetArrayName(i) for i in range(imageData.GetPointData().GetNumberOfArrays())]:
        if ia != arrayName:
            imageData.GetPointData().RemoveArray(ia)
    A = numpy_support.vtk_to_numpy(imageData.GetPointData().GetArray(arrayName))
    aArray = numpy_support.numpy_to_vtk(A.astype(np.int16), deep=1)
    aArray.SetName(arrayName)
    imageData.GetPointData().SetScalars(aArray)

    ## Set up writer and write.
    writer = vtk.vtkDICOMWriter()
    writer.SetInputData(imageData)
    writer.SetMetaData(meta)
    if TEMPORAL:
        writer.TimeAsVectorOn()
        writer.SetTimeSpacing(deltaT)
    # writer.SetMemoryRowOrderToFileNative()
    # writer.SetMemoryRowOrderToTopDown() # THESE LINES MAY BE VARIABLE...
    # writer.SetMemoryRowOrderToBottomUp() # THESE LINES MAY BE VARIABLE...
    # writer.SetFileSliceOrderToLHR()
    # writer.SetFileSliceOrderToSame()
    writer.SetGenerator(mrG)
    print(writer.GetFileSliceOrder())
    seriesDesc = getTag(reader, DicomTags.SeriesDescription)
    subDir = 'SE%d_%s_RECONSTRUCTED' % (newSeNum, seriesDesc)
    subDir = subDir.replace(' ', '_')
    outDir = os.path.join(outputDirectory, subDir)
    try:
        os.mkdir(outDir)
    except OSError:
        pass
    writer.SetFilePattern("%s/IM-" + newSeNumStr + "-%05d.dcm")
    writer.SetFilePrefix(outDir)
    writer.SetImageType("DERIVED/SECONDARY/MPR")
    writer.SetPatientMatrix(patientMatrix)
    writer.Write()
    return outDir


def writeJPEGsToDicoms(dicomTemplateFile, listOfJPEGFiles, outputDirectory, newSeNum=None):
    """
    :param dicomTemplateFile:
    :param listOfJPEGFiles:
    :param outputDirectory:
    :param newSeNum:
    :return:
    """

    def contrastStretch(array, minOutput, maxOutput, minOfInterest, maxOfInterest):
        output = (array - minOfInterest) * ((maxOutput - minOutput) / float(maxOfInterest - minOfInterest)) + minOutput
        output[output <= minOutput] = minOutput
        output[output >= maxOutput] = maxOutput
        return output

    import matplotlib.image as mpimg
    mrG = vtk.vtkDICOMMRGenerator()
    reader = getDicomReaderFromFileName(dicomTemplateFile)
    meta = reader.GetMetaData()
    ## Adjust the series number
    seriesStr = '%05d' % (newSeNum)
    i0, i1 = DicomTags.SeriesNumber
    tag = vtk.vtkDICOMTag(int(i0), int(i1))
    meta.SetAttributeValue(tag, seriesStr)

    ## Read images and set as int16
    L1 = True

    for iImage in listOfJPEGFiles:
        rr = vtk.vtkJPEGReader()
        rr.SetFileName(iImage)
        rgb = mpimg.imread(iImage)
        ii = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
        if L1:
            imageA3D = ii
            L1 = False
        else:
            imageA3D = np.vstack((imageA3D, ii))
    imageA3D = contrastStretch(imageA3D, 0, 15 ** 2, 0, 15 ** 2)
    imageA2D = imageA3D.flatten('C')
    aArray = numpy_support.numpy_to_vtk(imageA2D.astype(np.int16), deep=1)
    aArray.SetName('Scalars')

    print(imageA2D.shape)
    imageData = vtk.vtkImageData()
    imageData.SetOrigin(0.0, 0.0, 0.0)
    dims = [ii.shape[1], ii.shape[0], len(listOfJPEGFiles)]
    # dims = [ii.shape[1], ii.shape[0], 1]
    imageData.SetDimensions(dims)
    imageData.GetPointData().SetScalars(aArray)

    ## Set up writer and write.
    writer = vtk.vtkDICOMWriter()
    writer.SetInputData(imageData)
    writer.SetMetaData(meta)
    writer.SetMemoryRowOrderToFileNative()
    # writer.SetFileSliceOrderToRHR()
    writer.SetGenerator(mrG)
    seriesDesc = getTag(reader, DicomTags.SeriesDescription)
    subDir = 'SE%d_%s_RECONSTRUCTED' % (newSeNum, seriesDesc)
    subDir = subDir.replace(' ', '_')
    outDir = os.path.join(outputDirectory, subDir)
    try:
        os.mkdir(outDir)
    except OSError:
        pass
    writer.SetFilePattern("%s/IM-" + seriesStr + "-%05d.dcm")
    writer.SetFilePrefix(outDir)
    writer.SetImageType("DERIVED/SECONDARY/JPEG")
    # writer.SetPatientMatrix(patientMatrix)
    writer.Write()
    return outDir



    # if subsample > 0:
    #     data = fIO.readVTKFile(listOfVTKObjs_or_filePaths)
    #     extractGrid = vtk.vtkExtractVOI()
    #     extractGrid.SetInputData(data)
    #     extractGrid.SetSampleRate(subsample, subsample, subsample)
    #     extractGrid.IncludeBoundaryOn()
    #     extractGrid.Update()
    #     data = extractGrid.GetOutput()
    #     listOfVTKObjs_or_filePaths = fIO.writeVtkFile(data, listOfVTKObjs_or_filePaths[:-4] + '_SUB.vti')
    #     CLEAN_UP_LIST.append(listOfVTKObjs_or_filePaths)
# def vtkObjsToParaViewGlance(listOfVTKObj, outputFile, glanceHTML): # TODO - writer the issue - need to determine the writer
#     fileList = []
#     outputDir, fName = os.path.split(outputFile)
#     for k1, iObj in enumerate(listOfVTKObj):
#         fOut = fIO.writeVtkFile(iObj, os.path.join(outputDir, '%s_TEMP%d'%(fName, k1)))
#         fileList.append(fOut)
#     vtkFilesToParaViewGlance(fileList, outputFile, glanceHTML, QUIET=True, DEBUG=False)



def vtkFilesToParaViewGlance(listOfFilePaths, outputFile=None, glanceHtml=None, QUIET=False, DEBUG=False):
    """

    """
    CLEAN_UP_LIST = []
    FILE_TO_VTK_LIST = []
    ## --- Check inputs ---
    # Glance html
    if glanceHtml is None:
        thisDir = os.path.split(os.path.realpath(__file__))[0]
        glanceHtml = os.path.join(thisDir, 'ParaViewGlance.html')
        if not QUIET:
            print('Using ParaView glance file: %s'%(glanceHtml))
    if not os.path.isfile(glanceHtml):
        raise ValueError('%s does not exist'%(glanceHtml))

    ## --- Output file ---
    if outputFile is None:
        outputDir, fName = os.path.split(listOfFilePaths[0])
        fNameOut = os.path.splitext(fName)[0]+'.html'
        outputFile = os.path.join(outputDir, fNameOut)
    elif os.path.isdir(outputFile):
        outputDir = outputFile
        _, fName = os.path.split(listOfFilePaths[0])
        fNameOut = os.path.splitext(fName)[0]+'.html'
        outputFile = os.path.join(outputDir, fNameOut)
    else:
        outputDir, fNameOut = os.path.split(outputFile)

    ## --- VTK Objs / file paths ---
    for iPath in listOfFilePaths:
        if os.path.isfile(iPath):
            if iPath.endswith('nii'):
                iPath = nii2vti(iPath)
                CLEAN_UP_LIST.append(iPath)
            FILE_TO_VTK_LIST.append(iPath)
        else:
            if os.path.isdir(iPath): # If path to dicoms
                dcmToVTKPath = dicomDirToCinePVD(iPath, outputDir, fNameOut, RETURN_VTI=True)
                CLEAN_UP_LIST.append(dcmToVTKPath)
                FILE_TO_VTK_LIST.append(dcmToVTKPath)
            else:
                raise ValueError('%s does not exist' % (iPath))

    ## --- Build HTML Recursivly ---
    if not QUIET:
        print('Writing %s from base html %s, using:'%(outputFile, glanceHtml))
        for iFile in listOfFilePaths:
            print('    %s'%(iFile))

    for k1, iFile in enumerate(FILE_TO_VTK_LIST):
        outputTemp = outputFile[:-5]+'_TEMP%d.html'%(k1)
        glanceHtml = __vtkToHTML(iFile, glanceHtml, outputTemp)
        CLEAN_UP_LIST.append(outputTemp)
    os.rename(CLEAN_UP_LIST.pop(), outputFile)

    ## --- Clean up --- // Skipped if in DEBUG mode
    if not DEBUG:
        if not QUIET:
            print('Cleaning up:', str(CLEAN_UP_LIST))
        for ifile in CLEAN_UP_LIST:
            os.unlink(ifile)

    return outputFile


def __vtkToHTML(vtkDataPath, glanceHtml, outputFile):
    # Extract data as base64
    with open(vtkDataPath, "rb") as data:
        dataContent = data.read()
        base64Content = base64.b64encode(dataContent)
        base64Content = base64Content.decode().replace("\n", "")
    # Create new output file
    with open(glanceHtml, mode="r", encoding="utf-8") as srcHtml:
        with open(outputFile, mode="w", encoding="utf-8") as dstHtml:
            for line in srcHtml:
                if "</body>" in line:
                    dstHtml.write("<script>\n")
                    dstHtml.write('var contentToLoad = "%s";\n\n' % base64Content)
                    dstHtml.write(
                        'Glance.importBase64Dataset("%s" , contentToLoad, glanceInstance.proxyManager);\n'
                        % os.path.basename(vtkDataPath)
                    )
                    dstHtml.write("glanceInstance.showApp();\n")
                    dstHtml.write("</script>\n")
                dstHtml.write(line)
    return outputFile

## NOT THE WAY TO DO IT
# def __buildRawImageData(dims, res, origin):
#     newImg = vtk.vtkImageData()
#     newImg.SetSpacing(res[0] ,res[1] ,res[2])
#     newImg.SetOrigin(origin[0], origin[1], origin[2])
#     newImg.SetDimensions(dims[0] ,dims[1] ,dims[2])
#     return newImg

# def dicomDirToRAW_VTKImageData(dicomDirectory):
#     dcmReader = getDicomReaderFromDirectory(dicomDirectory) # this takes care of sorting
#     # return dcmReader.GetOutput()
#     res = getResolution(dcmReader)
#     dims = getDimensions(dcmReader)
#     origin = getImagePositionPatient(dcmReader)
#     return __buildRawImageData(dims, res, origin)



if __name__ == '__main__':

    # --------------------------------------------------------------------------
    #  ARGUMENT PARSING
    # --------------------------------------------------------------------------
    ap = argparse.ArgumentParser(description='Dicoms to vtk')

    groupS = ap.add_argument_group('Subject')
    groupS.add_argument('-i', dest='input', help='input files or directories of dicoms or vtk (for vtk2html)', nargs='*', type=str, required=True)
    groupS.add_argument('-ALL_SUB', dest='ALL_SUB', help='write out per subdirectory', action='store_true')
    groupS.add_argument('-o', dest='outputDir', help='output directory', type=str, required=True)
    groupS.add_argument('-n', dest='outputName', help='output name', type=str, required=True)

    groupP = ap.add_argument_group('Parameter')
    groupP.add_argument('-se', dest='series_numbers', help='Series numbers', nargs='*', type=int, default=[])
    groupP.add_argument('-C', dest='CINE', help='CINE', action='store_true')
    groupP.add_argument('-CI', dest='CINE_VTI', help='CINE VTI output', action='store_true')
    groupP.add_argument('-FDF', dest='FDF', help='FourDFlow', action='store_true')
    groupP.add_argument('-HTML', dest='HTML', help='Write dicoms to 3D html', action='store_true')
    # groupP.add_argument('-html_sub', dest='html_sub', help='Level to subsample data for HTML (default: 0)', type=int, default=0)
    groupP.add_argument('-DEBUG', dest='DEBUG', help='DEBUG', action='store_true')
    groupP.add_argument('-QUIET', dest='QUIET', help='QUIET', action='store_true')

    args = ap.parse_args()
    print('Started: %s' % (datetime.datetime.strftime(datetime.datetime.now(), '%H:%M:%S')))
    if len(args.series_numbers) > 0:
        args.input = seriesNumbersToDicomDirList(args.input[0], args.series_numbers)
    print(args.input)
    if args.CINE or args.CINE_VTI:
        if args.ALL_SUB:
            for iDir in os.listdir(args.input[0]):
                subDir = os.path.join(args.input[0], iDir)
                dicomDirToCinePVD(subDir, args.outputDir, iDir+'_'+args.outputName, args.CINE_VTI)
        else:
            dicomDirToCinePVD(args.input[0], args.outputDir, args.outputName, args.CINE_VTI)
    elif args.FDF:
        dicomDirToFDFPVD(args.input, args.outputDir, args.outputName, QUIET=args.QUIET, DEBUG=args.DEBUG)
    elif args.HTML:
        vtkFilesToParaViewGlance(args.input, os.path.join(args.outputDir, args.outputName), QUIET=args.QUIET, DEBUG=args.DEBUG)

    print('Finished: %s' % (datetime.datetime.strftime(datetime.datetime.now(), '%H:%M:%S')))

