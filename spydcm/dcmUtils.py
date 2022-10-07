# -*- coding: utf-8 -*-

"""This module has high level dicom operations accessed by classes in dcmTK.py
"""

import os
import pydicom as dicom
import tarfile
from tqdm import tqdm 
import numpy as np

import spydcm.dcmTools as dcmTools


def getDicomDictFromTar(tarFileToRead, QUIET=True, FORCE_READ=False, FIRST_ONLY=False, OVERVIEW_ONLY=True,
                        matchingTagValuePair=None):
    # for sub dir in tar get first dicom - return list of ds
    dsDict = {}
    if tarFileToRead.endswith('gz'):
        tar = tarfile.open(tarFileToRead, "r:gz")
    else:
        tar = tarfile.open(tarFileToRead)
    successReadDirs = set()
    for member in tar:
        if member.isfile():
            root = os.path.split(member.name)[0]
            if FIRST_ONLY and (root in successReadDirs):
                continue
            thisFile=tar.extractfile(member)
            try:
                dataset = dicom.read_file(thisFile, stop_before_pixels=OVERVIEW_ONLY, force=FORCE_READ)#, specific_tags=['StudyInstanceUID','SeriesInstanceUID'])
                if matchingTagValuePair is not None:
                    if dataset.get(matchingTagValuePair[0], 'NIL') != matchingTagValuePair[1]:
                        continue
                studyUID = str(dataset.StudyInstanceUID)
                seriesUID = str(dataset.SeriesInstanceUID)
                if studyUID not in dsDict:
                    dsDict[studyUID] =  {}
                if seriesUID not in dsDict[studyUID]:
                    dsDict[studyUID][seriesUID] = []
                dsDict[studyUID][seriesUID].append(dataset)

                if FIRST_ONLY:
                    successReadDirs.add(root)
            except dicom.filereader.InvalidDicomError:
                if not QUIET:
                    print('FAIL: %s'%(thisFile))
    tar.close()
    return dsDict

def anonymiseDicomDS(dataset, anon_birthdate=True, remove_private_tags=True, anonName=None):
    # Define call-back functions for the dataset.walk() function
    def PN_callback(ds, data_element):
        """Called from the dataset "walk" recursive function for all data elements."""
        if data_element.VR == "PN":
            data_element.value = 'anonymous'
        if data_element.name == "Institution Name":
            data_element.value = 'anonymous'
        if (anonName is not None) & (data_element.name == "Patient's Name"):
            data_element.value = anonName
    # Remove patient name and any other person names
    dataset.walk(PN_callback)
    # Change ID
    dataset.PatientID = ''
    # Remove data elements (should only do so if DICOM type 3 optional)
    # Use general loop so easy to add more later
    # Could also have done: del ds.OtherPatientIDs, etc.
    for name in ['OtherPatientIDs', 'OtherPatientIDsSequence']:
        if name in dataset:
            delattr(dataset, name)
    if anon_birthdate:
        for name in ['PatientBirthDate']:
            if name in dataset:
                dataset.data_element(name).value = ''
    # Remove private tags if function argument says to do so.
    if remove_private_tags:
        dataset.remove_private_tags()
    return dataset

def getSaveFileNameFor_ds_UID(ds, outputRootDir):
    destFile = os.path.join(outputRootDir, ds.PatientID, ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID)
    return destFile

def __getDSSaveFileName(ds):
    try:
        return 'IM-%05d-%05d.dcm'%(int(ds.SeriesNumber),
                                        int(ds.InstanceNumber))
    except (TypeError, KeyError, AttributeError):
        return 'IM-%s.dcm'%(ds.SOPInstanceUID)
        # try:
        #     return 'IM-%s.dcm'%(ds.SOPInstanceUID)
        # except AttributeError:
        #     return 'IM-%s.dcm'%(str(uuid.uuid1()))

def writeOut_ds(ds, outputRootDir, anonName=None, WRITE_LIKE_ORIG=True):
    destFile = os.path.join(outputRootDir, __getDSSaveFileName(ds))
    os.makedirs(outputRootDir, exist_ok=True)
    if anonName is not None:
        try:
            ds = anonymiseDicomDS(ds, anonName=anonName)
        except NotImplementedError:
            pass
    ds.save_as(destFile, write_like_original=WRITE_LIKE_ORIG)
    return destFile


def organiseDicomHeirachyByUIDs(rootDir, HIDE_PROGRESSBAR=False, FORCE_READ=False, ONE_FILE_PER_DIR=False, OVERVIEW=True):
    dsDict = {}
    successReadDirs = set()
    nFiles = dcmTools.__countFilesInDir(rootDir)
    for thisFile in tqdm(dcmTools.walkdir(rootDir), total=nFiles, leave=True, disable=HIDE_PROGRESSBAR):
        if 'dicomdir' in os.path.split(thisFile)[1].lower():
            continue
        if thisFile.endswith('json'):
            continue
        thisDir, ff = os.path.split(thisFile)
        if ONE_FILE_PER_DIR:
            if thisDir in successReadDirs:
                continue
        try:
            dataset = dicom.read_file(thisFile, specific_tags=['StudyInstanceUID','SeriesInstanceUID'], stop_before_pixels=OVERVIEW, force=FORCE_READ)
            studyUID = str(dataset.StudyInstanceUID)
            seriesUID = str(dataset.SeriesInstanceUID)
            if studyUID not in dsDict:
                dsDict[studyUID] =  {}
            if seriesUID not in dsDict[studyUID]:
                dsDict[studyUID][seriesUID] = []
            dsDict[studyUID][seriesUID].append(dataset)
            successReadDirs.add(thisDir)
        except dicom.filereader.InvalidDicomError:
            # print('Error reading %s'%(thisFile))
            continue
    return dsDict



def getAllDirsUnderRootWithDicoms(rootDir, QUIET=True, FORCE_READ=False):
    fullDirsWithDicoms = []
    for root, dirs, files in os.walk(rootDir):
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
        writeOut_ds(ds, outputDir)

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


