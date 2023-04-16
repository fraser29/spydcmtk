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
from spydcmtk.defaults import MANUSCRIPT_TABLE_TAG_LIST



def writeDirectoryToNII(dcmDir, outputPath, fileName):
    dcm2niiCmd = "dcm2nii -p n -e y -d n -x n -o '%s' '%s'"%(outputPath, dcmDir)
    print('RUNNING: %s'%(dcm2niiCmd))
    os.system(dcm2niiCmd)
    list_of_files = glob.glob(os.path.join(outputPath, '*.nii.gz'))  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    newFileName = os.path.join(outputPath, fileName)
    os.rename(latest_file, newFileName)
    print('Made %s --> as %s'%(latest_file, newFileName))


def buildTableOfDicomParamsForManuscript(topLevelDirectoryList, seriesDescriptionIdentifier, tagList=MANUSCRIPT_TABLE_TAG_LIST):
    dd = {}
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
                    dd.update({iSeries.getSeriesOutDirName(): iSeries.getSeriesInfoDict()})         
    
        # for dcmSE in dcmStudy:
        #     seInfoList.append(dcmSE.getSeriesInfoDict())
        # df = pd.DataFrame(data=seInfoList)
        # df.to_csv(self.getSeriesMetaCSV())


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

def writeNumpyArrayToDicom(pixelArray, dcmTemplate_or_ds, patientMatrixDict, outputDir, tagUpdateDict=None):
    """
    patientMatrixDict = {PixelSpacing: [1,2], ImagePositionPatient: [1,3], ImageOrientationPatient: [1,6], SliceThickness: 1}
    Note - use "SliceThickness" in patientMatrixDict - with assumption that SliceThickness==SpacingBetweenSlices (explicitely built htis way - ndArray can not have otherwise)
    """
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
