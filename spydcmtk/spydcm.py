# -*- coding: utf-8 -*-

"""Module that exposes the routines and utilities making up SPYDCMTK
"""

import os
import glob
import numpy as np
import pydicom as dicom

# Local imports 
import spydcmtk.dcmTools as dcmTools
import spydcmtk.dcmTK as dcmTK



def writeDirectoryToNII(dcmDir, outputPath, fileName):
    dcm2niiCmd = "dcm2nii -p n -e y -d n -x n -o '%s' '%s'"%(outputPath, dcmDir)
    print('RUNNING: %s'%(dcm2niiCmd))
    os.system(dcm2niiCmd)
    list_of_files = glob.glob(os.path.join(outputPath, '*.nii.gz')) 
    latest_file = max(list_of_files, key=os.path.getctime)
    newFileName = os.path.join(outputPath, fileName)
    os.rename(latest_file, newFileName)
    print('Made %s --> as %s'%(latest_file, newFileName))


def buildTableOfDicomParamsForManuscript(topLevelDirectoryList, seriesDescriptionIdentifier):
    dfData = []
    for inspectDir in topLevelDirectoryList:
        try:
            ssL = dcmTK.ListOfDicomStudies.setFromDirectory(inspectDir, OVERVIEW=True, HIDE_PROGRESSBAR=True,
                                            ONE_FILE_PER_DIR=True)
        except IndexError: # no dicoms found 
            continue
        for ss in ssL:
            matchingSeries = ss.getSeriesMatchingDescription([seriesDescriptionIdentifier], RETURN_SERIES_OBJ=True)
            if matchingSeries is not None:
                for iSeries in matchingSeries:
                    resDict = iSeries.getSeriesInfoDict()
                    resDict["Identifier"] = dcmTools.getDicomFileIdentifierStr(iSeries[0])
                    dfData.append(resDict)
    stats = {}
    tagList = sorted(dfData[0].keys())
    print(','+','.join(tagList))
    for row in dfData:
        print(',', end='')
        for i in tagList:
            print(f'{row[i]},', end='')
            stats.setdefault(i, []).append(row[i])
        print('', end='\n')
    print('\n\nSTATS:', end='\n')
    for label, myFunc in zip(['Mean', 'Standard Deviation', 'Median', 'Min', 'Max'], 
                             [np.mean, np.std, np.median, np.min, np.max]):
        for k1 in range(len(tagList)):
            if k1 == 0:
                print(f'{label},', end='')
            try:
                print(f'{myFunc(stats[tagList[k1]])},', end='')
            except:
                print('NAN,', end='')
        print('', end='\n')


def getAllDirsUnderRootWithDicoms(rootDir, QUIET=True, FORCE_READ=False):
    fullDirsWithDicoms = []
    for root, _, files in os.walk(rootDir):
        for iFile in files:
            thisFile = os.path.join(root, iFile)
            try:
                dicom.read_file(thisFile, stop_before_pixels=True, defer_size=16, force=FORCE_READ) # will error if not dicom
                if not QUIET:
                    print('OK: %s'%(thisFile))
                fullDirsWithDicoms.append(root)
                break
            except dicom.filereader.InvalidDicomError:
                if not QUIET:
                    print('FAIL: %s'%(thisFile))
                continue
    return fullDirsWithDicoms


def writeVTIToDicoms(vtiFile, dcmTemplateFile_or_ds, outputDir, arrayName=None, tagUpdateDict=None, patientMatrixDict=None):
    if type(vtiFile) == str:
        vti = dcmTK.dcmVTKTK.readVTKFile(vtiFile)
    if arrayName is None:
        A = dcmTK.dcmVTKTK.getScalarsAsNumpy(vti)
    else:
        A = dcmTK.dcmVTKTK.getArrayAsNumpy(vti, arrayName)
    A = np.reshape(A, vti.GetDimensions(), 'F')
    if patientMatrixDict is None:
        patientMatrixDict = dcmTK.dcmVTKTK.getPatientMatrixDict(vti)
    return writeNumpyArrayToDicom(A, dcmTemplateFile_or_ds, patientMatrixDict, outputDir, tagUpdateDict=tagUpdateDict)


def writeNumpyArrayToDicom(pixelArray, dcmTemplate_or_ds, patientMatrixDict, outputDir, tagUpdateDict=None):
    """
    patientMatrixDict = {PixelSpacing: [1,2], ImagePositionPatient: [1,3], ImageOrientationPatient: [1,6], SliceThickness: 1}
    Note - use "SliceThickness" in patientMatrixDict - with assumption that SliceThickness==SpacingBetweenSlices (explicitely built this way - ndArray can not have otherwise)
    """
    print(patientMatrixDict)
    if tagUpdateDict is None:
        tagUpdateDict = {}
    if type(dcmTemplate_or_ds) == str:
        ds = dicom.read_file(dcmTemplate_or_ds)
    else:
        ds = dcmTemplate_or_ds
    nRow, nCol, nSlice = pixelArray.shape
    if pixelArray.dtype != np.int16:
        pixelArray = pixelArray.astype(np.int16)
    SeriesUID = dicom.uid.generate_uid()
    try:
        SeriesNumber = tagUpdateDict['SeriesNumber']
    except KeyError:
        SeriesNumber = ds.SeriesNumber * 100
    for k in range(nSlice):
        sliceA = pixelArray[:,:,k]
        ds.SeriesInstanceUID = SeriesUID
        # ds.MediaStorageSOPInstanceUID = dicom.uid.generate_uid()
        ds.SOPInstanceUID = dicom.uid.generate_uid()
        ds.Rows = nRow
        ds.Columns = nCol
        ds.ImagesInAcquisition = nSlice
        ds.InStackPositionNumber = k+1
        ds.RawDataRunNumber = k+1
        ds.SeriesNumber = SeriesNumber
        ds.InstanceNumber = k+1
        ds.SliceThickness = patientMatrixDict['SliceThickness']
        ds.SpacingBetweenSlices = patientMatrixDict['SliceThickness']
        ds.SmallestImagePixelValue = max([0, np.min(sliceA)])
        mx = min([32767, np.max(sliceA)])
        ds.LargestImagePixelValue = mx
        ds.WindowCenter = int(mx / 2)
        ds.WindowWidth = int(mx / 2)
        ds.PixelSpacing = list(patientMatrixDict['PixelSpacing'])
        kVec = np.cross(patientMatrixDict['ImageOrientationPatient'][:3],
                        patientMatrixDict['ImageOrientationPatient'][3:])
        ImagePositionPatient = np.array(patientMatrixDict['ImagePositionPatient']) + k*kVec*patientMatrixDict['SliceThickness']
        ds.ImagePositionPatient = list(ImagePositionPatient)
        try:
            sliceLoc = tagUpdateDict['SliceLocation0'] + k*patientMatrixDict['SliceThickness']
        except KeyError:
            sliceLoc = dcmTools.distPts(ImagePositionPatient, np.array(patientMatrixDict['ImagePositionPatient']))
        ds.SliceLocation = sliceLoc
        # ds.ImageLocation = k+1
        ds.ImageOrientationPatient = list(patientMatrixDict['ImageOrientationPatient'])
        # for iKey in tagUpdateDict.keys():
        #     ds[iKey] = tagUpdateDict[iKey]
        ds.PixelData = sliceA.tostring()
        dcmTools.writeOut_ds(ds, outputDir)


def returnFirstDicomFound(rootDir, FILE_NAME_ONLY=False):
    """
    Search recursively for first dicom file under root and return.
    If have dicoms in nice folder structure then this can be a fast way to find, e.g. all series with protocol X

    :param rootDir: directory on filesystem
    :param FILE_NAME_ONLY: If true will return the file name [Default False]
    :return: pydicom dataset<without pixel data> or fileName<str>
    """
    for root, _, files in os.walk(rootDir):
        for iFile in files:
            if 'dicomdir' in iFile.lower():
                continue
            thisFile = os.path.join(root, iFile)
            try:
                dataset = dicom.read_file(thisFile, stop_before_pixels=True)
                if FILE_NAME_ONLY:
                    return thisFile
                else:
                    return dataset
            except dicom.filereader.InvalidDicomError:
                continue
    return None

def directoryToVTI(dcmDirectory, outputFolder, QUITE=True, FORCE=False):
    outputFiles = []
    ListDicomStudies = dcmTK.ListOfDicomStudies.setFromInput(dcmDirectory, HIDE_PROGRESSBAR=QUITE, FORCE_READ=FORCE, OVERVIEW=False) 
    for iDS in ListDicomStudies:
        for iSeries in iDS:
            fOut = iSeries.writeToVTI(outputPath=outputFolder, outputNaming=['PatientName', 'SeriesNumber', 'SeriesDescription'])
            outputFiles.append(fOut)
    return outputFiles
