# -*- coding: utf-8 -*-

"""Classes for working with Dicom studies
"""
import copy
import os
import pydicom as dicom
from pydicom.uid import generate_uid
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np
import shutil

# Local imports 
import spydcmtk.dcmTools as dcmTools
import spydcmtk.dcmVTKTK as dcmVTKTK
from spydcmtk.helpers import SERIES_OVERVIEW_TAG_LIST, STUDY_OVERVIEW_TAG_LIST, SUBJECT_OVERVIEW_TAG_LIST




## =====================================================================================================================
##        CLASSES
## =====================================================================================================================
class DicomSeries(list):
    """
    Extends a list of ds (pydicom dataset) objects.
    """
    def __init__(self, dsList=None, OVERVIEW=False, HIDE_PROGRESSBAR=False, FORCE_READ=False, SAFE_NAME_MODE=False):
        """
        Set OVERVIEW = False to read pixel data as well (at a cost)
        """
        if dsList is None:
            dsList = []
        self.OVERVIEW = OVERVIEW
        self.HIDE_PROGRESSBAR = HIDE_PROGRESSBAR
        self.FORCE_READ = FORCE_READ
        self.SAFE_NAME_MODE = SAFE_NAME_MODE
        list.__init__(self, dsList)

    def __str__(self):
        return ' '.join([str(i) for i in self.getSeriesOverview()[1]])

    @classmethod
    def setFromDirectory(cls, dirName, OVERVIEW=False, HIDE_PROGRESSBAR=False, FORCE_READ=False, ONE_FILE_PER_DIR=False):
        dicomDict = dcmTools.organiseDicomHeirachyByUIDs(dirName, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ, ONE_FILE_PER_DIR=ONE_FILE_PER_DIR, OVERVIEW=OVERVIEW)
        dStudyList = ListOfDicomStudies.setFromDcmDict(dicomDict, OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ)
        if len(dStudyList) > 1:
            raise ValueError('More than one study found - use ListOfDicomStudies class')
        dStudy = dStudyList[0]
        if len(dStudy) > 1:
            raise ValueError('More than one series found - use DicomStudy class')
        return cls(dStudy[0], OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ)

    @classmethod
    def setFromFileList(cls, fileList, OVERVIEW=False, HIDE_PROGRESSBAR=False, FORCE_READ=False):
        dsList = [dicom.read_file(i, stop_before_pixels=OVERVIEW, force=FORCE_READ) for i in fileList]
        return cls(dsList, OVERVIEW, HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ)

    def getRootDir(self):
        return os.path.split(self[0].filename)[0]

    def sortByInstanceNumber(self):
        self.sort(key=dcmTools.instanceNumberSortKey)

    def sortBySlice_InstanceNumber(self):
        self.sort(key=dcmTools.sliceLoc_InstanceNumberSortKey)

    def getTag(self, tag, dsID=0, ifNotFound='Unknown', convertToType=None):
        try:
            tt = self.getTagObj(tag, dsID)
            if convertToType is not None:
                return convertToType(tt.value)
            return tt.value
        except KeyError:
            return ifNotFound

    def setTag(self, tag, value):
        for ds in self:
            ds.data_element(tag).value = value

    def getTagObj(self, tag, dsID=0):
        try:
            if tag[:2] == '0x':
                tag = int(tag, 16)
        except TypeError:
            pass # Not string
        return self[dsID][tag]

    def getTagValuesList(self, tagList, RETURN_STR):
        """
        Build table with file name and value from each tag in list.
        Return list of lists
        """
        valueList = []
        for dsID, i in enumerate(self):
            subList = ['"%s"'%(str(i.filename))] + [str(self.getTag(t, dsID=dsID)) for t in tagList]
            valueList.append(subList)
        if RETURN_STR:
            return dcmTools._tagValuesListToString(tagList, valueList)
        return valueList

    def getTagListAndNames(self, tagList, dsID=0):
        names, vals = [], []
        for i in tagList:
            try:
                dataEle = self.getTagObj(i, dsID=dsID)
                iname = str(dataEle.keyword)
                if len(iname) == 0:
                    iname = str(dataEle.name)
                    iname = iname.replace('[','').replace(']','').replace(' ','')
                names.append(iname)
                vals.append(str(dataEle.value))
            except KeyError:
                names.append(str(i))
                vals.append('Unknown')
        return names, vals

    def tagsToJson(self, jsonFileOut):
        dOut = {}
        for ds in self:
            dd = ds.to_json_dict()
            dd.pop('7FE00010') # Remove actual PixelData
            dOut[ds.InstanceNumber] = dd
        dcmTools.writeDictionaryToJSON(jsonFileOut, dOut)

    def getSeriesOverview(self, tagList=SERIES_OVERVIEW_TAG_LIST):
        names, vals = self.getTagListAndNames(tagList)
        names.append('ImagesInSeries')
        vals.append(len(self))
        return names, vals

    def getSeriesTimeAsDatetime(self):
        dos = self.getTag('SeriesDate', ifNotFound="19000101")
        tos = self.getTag('SeriesTime', ifNotFound="000000")
        try:
            return datetime.strptime(f"{dos} {tos}", "%Y%m%d %H%M%S.%f")
        except ValueError:
            return datetime.strptime(f"{dos} {tos}", "%Y%m%d %H%M%S")

    def getImagePositionPatient_np(self, dsID):
        ipp = self.getTag('ImagePositionPatient', dsID=dsID)
        return np.array(ipp)

    def getImageOrientationPatient_np(self, dsID=0):
        iop = self.getTag('ImageOrientationPatient', dsID=dsID)
        return np.array(iop)

    def yieldDataset(self):
        for ds in self:
            if self.OVERVIEW:
                yield  dicom.read_file(ds.filename, stop_before_pixels=False, force=True)
            else:
                yield ds

    def isCompressed(self):
        return dcmTools._isCompressed(self[0])

    def getSeriesNumber(self):
        return int(self.getTag('SeriesNumber', ifNotFound=0))

    def getSeriesOutDirName(self, SE_RENAME={}):
        thisSeNum = self.getTag('SeriesNumber', ifNotFound='#')
        if thisSeNum in SE_RENAME.keys():
            return SE_RENAME[thisSeNum]
        return dcmTools.cleanString('SE%s_%s'%(self.getTag('SeriesNumber', ifNotFound='#'),
                                                    self.getTag('SeriesDescription', ifNotFound='SeriesDesc-unknown')))

    def resetUIDs(self, studyUID=None):
        """
            Reset UIDs - helpful for case of anon for analysis - but going back into PACS 
        """
        if studyUID is None:
            studyUID = str(generate_uid())
        seriesUID = str(generate_uid())
        for k1 in range(len(self)):
            self[k1].SOPInstanceUID = str(generate_uid())
            self[k1].SeriesInstanceUID = seriesUID
            self[k1].StudyInstanceUID = studyUID


    def writeToOrganisedFileStructure(self, studyOutputDir, anonName=None, anonID='', UIDupdateDict={}, SE_RENAME={}, LIKE_ORIG=True, SAFE_NAMING_CHECK=True):
        """ Recurse down directory tree - grab dicoms and move to new
            hierarchical folder structure
            SE_RENAME = dict of SE# and Name to rename the SE folder    
            LIKE_ORIG - set to False if updated some tags
        """
        if SAFE_NAMING_CHECK:
            self.checkIfShouldUse_SAFE_NAMING()
        ADD_TRANSFERSYNTAX = False
        LIKE_ORIG = True
        destFile = None
        seriesOutDirName = self.getSeriesOutDirName(SE_RENAME)
        seriesOutputDir = os.path.join(studyOutputDir, seriesOutDirName)
        if self.FORCE_READ:
            ADD_TRANSFERSYNTAX = True ## THIS IS A BESPOKE CHANGE
        TO_DECOMPRESS = self.isCompressed()
        UIDupdateDict['SeriesInstanceUID'] = dicom.uid.generate_uid()
        for ds in tqdm(self.yieldDataset(), total=len(self), disable=self.HIDE_PROGRESSBAR):
            if TO_DECOMPRESS:
                ds.decompress('gdcm')
            if ADD_TRANSFERSYNTAX:
                ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
                LIKE_ORIG=False
            destFile = dcmTools.writeOut_ds(ds, seriesOutputDir, anonName, anonID, UIDupdateDict, WRITE_LIKE_ORIG=LIKE_ORIG, SAFE_NAMING=self.SAFE_NAME_MODE)
        return destFile

    def __generateFileName(self, tagsToUse, extn):
        if type(tagsToUse) == str:
            fileName = tagsToUse
        else:
            fileName = '_'.join([str(self.getTag(i)) for i in tagsToUse])
        fileName = dcmTools.cleanString(fileName)
        if (len(extn) > 0) and (not extn.startswith('.')):
            extn = '.'+extn
        return fileName+extn

    def writeToNII(self, outputPath, outputNamingTags=('PatientName', 'SeriesNumber', 'SeriesDescription')):
        fileName = self.__generateFileName(outputNamingTags, '.nii.gz')
        return dcmTools.writeDirectoryToNII(self.getRootDir(), outputPath, fileName=fileName)

    def writeToVTI(self, outputPath, outputNamingTags=('PatientName', 'SeriesNumber', 'SeriesDescription'), INCLUDE_MATRIX=True):
        fileName = self.__generateFileName(outputNamingTags, '')
        vtiDict = self.buildVTIDict(INCLUDE_MATRIX=INCLUDE_MATRIX)
        return dcmVTKTK.writeVTIDict(vtiDict, outputPath, fileName)

    def writeToVTS(self, outputPath, outputNamingTags=('PatientName', 'SeriesNumber', 'SeriesDescription')):
        vtsDict = self.buildVTSDict()
        fileName = self.__generateFileName(outputNamingTags, '')
        return dcmVTKTK.writeVtkPvdDict(vtsDict, outputPath, filePrefix=fileName, fileExtn='vts', BUILD_SUBDIR=True)

    def buildVTSDict(self):
        vtiDict = self.buildVTIDict(INCLUDE_MATRIX=False)
        vtsDict = {}
        for ikey in vtiDict.keys():
            vtsDict[ikey] = dcmVTKTK.vtiToVts_viaTransform(vtiDict[ikey])
        return vtsDict

    def buildVTIDict(self, INCLUDE_MATRIX=True):
        A, meta = self.getPixelDataAsNumpy()
        return dcmVTKTK.arrToVTI(A, meta, self[0], INCLUDE_MATRIX=INCLUDE_MATRIX)

    @property
    def sliceLocations(self):
        return sorted([float(self.getTag('SliceLocation', i, ifNotFound=0.0)) for i in range(len(self))])

    def getNumberOfSlicesPerVolume(self):
        sliceLoc = self.sliceLocations
        return len(set(sliceLoc))

    def getNumberOfTimeSteps(self):
        sliceLoc = self.sliceLocations
        sliceLocS = set(sliceLoc)
        return sliceLoc.count(sliceLocS.pop())

    def getMatrix(self):
        self.sortBySlice_InstanceNumber()
        ipp = self.getImagePositionPatient_np(0)
        iop = self.getImageOrientationPatient_np()
        vecC = np.cross(iop[3:6], iop[:3])
        dx, dy, dz = self.getDeltaCol(), self.getDeltaRow(), self.getDeltaSlice()
        return np.array([[iop[3]*dx, iop[0]*dy, vecC[0]*dz, iop[0]],
                         [iop[4]*dx, iop[1]*dy, vecC[1]*dz, iop[1]],
                         [iop[5]*dx, iop[2]*dy, vecC[2]*dz, iop[2]],
                         [0,0,0,1]])

    def getPixelDataAsNumpy(self):
        """Get pixel data as numpy array organised by slice and time(if present).
            Also return dictionary of meta ('Spacing', 'Origin', 'ImageOrientationPatient', 'Times')
        """
        I,J,K = int(self.getTag('Columns')), int(self.getTag('Rows')), int(self.getNumberOfSlicesPerVolume())
        self.sortBySlice_InstanceNumber()
        N = self.getNumberOfTimeSteps()
        A = np.zeros((I, J, K, N))
        c0 = 0
        for k1 in range(K):
            for k2 in range(N):
                iA = self[c0].pixel_array.T
                A[:, :, k1, k2] = iA
                c0 += 1
        dt = self.getTemporalResolution()
        meta = {'Spacing':[self.getDeltaCol()*0.001, self.getDeltaRow()*0.001, self.getDeltaSlice()*0.001], 
                'Origin': [i*0.001 for i in self.getImagePositionPatient_np(0)], 
                'ImageOrientationPatient': self.getTag('ImageOrientationPatient'), 
                'Times': [dt*n*0.001 for n in range(N)]}
        return A, meta

    def getScanDuration_secs(self):
        try:
            return self.getTag(0x0019105a, ifNotFound=0.0) / 1000000.0
        except AttributeError:
            return 0.0

    def _getPixelSpacing(self):
        return [float(i) for i in self.getTag('PixelSpacing', ifNotFound=[0.0,0.0])]

    def getDeltaRow(self):
        return self._getPixelSpacing()[0]

    def getDeltaCol(self):
        return self._getPixelSpacing()[1]

    def getDeltaSlice(self):
        self.sortByInstanceNumber()
        p0 = self.getImagePositionPatient_np(0)
        sliceLoc = [distBetweenTwoPts(p0, self.getImagePositionPatient_np(i)) for i in range(len(self))]
        sliceLocS = sorted(list(set(sliceLoc)))
        if len(sliceLocS) == 1: # May be CINE at same location
            dZ = self.getTag('SpacingBetweenSlices', ifNotFound=None)
            if dZ is None:
                dZ = self.getTag('SliceThickness')
            return float(dZ)
        return np.mean(np.diff(sliceLocS))

    def getTemporalResolution(self):
        try:
            return float(self.getTag('NominalInterval', ifNotFound=0.0)/self.getTag('CardiacNumberOfImages', ifNotFound=1))
        except ZeroDivisionError:
            return 0       

    def getManufacturer(self):
        return self.getTag(0x00080070, ifNotFound='Unknown')
    def IS_GE(self):
        return self.getManufacturer().lower().startswith('ge')
    def IS_SIEMENS(self):
        return self.getManufacturer().lower().startswith('siemens')

    def getPulseSequenceName(self):
        if self.IS_GE():
            return self.getTag(0x0019109c)
        else:
            return self.getTag(0x00180024)

    def getInternalPulseSequenceName(self):
        return self.getTag(0x0019109e)

    def getSeriesDescription(self):
        return self.getTag('SeriesDescription')

    def getSeriesInfoDict(self):
        outDict = {'SeriesNumber':self.getTag('SeriesNumber'),
            'SeriesDescription':self.getTag('SeriesDescription'),
            'StudyDate':self.getTag('StudyDate'),
            'StartTime':self.getTag('AcquisitionTime'),
            'ScanDuration':self.getScanDuration_secs(),
            'nTime':self.getTag('CardiacNumberOfImages'),
            'nRow':self.getTag('Rows'),
            'nCols':self.getTag('Columns'),
            'dRow':self.getDeltaRow(),
            'dCol':self.getDeltaCol(),
            'dSlice':self.getTag('SliceThickness'),
            'dTime': self.getTemporalResolution(),
            'SpacingBetweenSlices':self.getTag('SpacingBetweenSlices'),
            'FlipAngle':self.getTag('FlipAngle'),
            'InPlanePhaseEncodingDirection':self.getTag('InPlanePhaseEncodingDirection'),
            'HeartRate':self.getTag('HeartRate'),
            'EchoTime':self.getTag('EchoTime'),
            'RepetitionTime':self.getTag('RepetitionTime'),
            'PulseSequenceName':self.getPulseSequenceName(),
            'MagneticFieldStrength': self.getTag('MagneticFieldStrength'),
            'InternalPulseSequenceName':self.getInternalPulseSequenceName(),
            'ReconstructionDiameter': self.getTag(0x00181100),
            'AcquisitionMatrix': f'"{str(self.getTag(0x00181310))}"',
            'ManufacturerModelName': self.getTag('ManufacturerModelName'),
            'SoftwareVersions': f'"{str(self.getTag(0x00181020))}"',}
        outDict['nSlice'] = len(self)
        try:
            outDict['AcquiredResolution'] = float(outDict['ReconstructionDiameter']) / float(max(self.getTag(0x00181310)))
        except ValueError:
            outDict['AcquiredResolution'] = f"{outDict['dRow']},{outDict['dCol']}"
        return outDict

    def checkIfShouldUse_SAFE_NAMING(self, se_instance_set=None):
        if se_instance_set is None:
            se_instance_set = set()
        for k1 in range(len(self)):
            se = self.getTag('SeriesNumber', dsID=k1, ifNotFound='unknown')
            instance = self.getTag('InstanceNumber', dsID=k1, ifNotFound='unknown')
            se_instance_str = f"{se}_{instance}"
            if se_instance_str in se_instance_set:
                self.SAFE_NAMING = True
                break
            se_instance_set.add(se_instance_str)

    def getStudyOutputDir(self, rootDir='', anonName=None, studyPrefix=''):
        # 'AccessionNumber' 'StudyID'
        if anonName is not None:
            ss = dcmTools.cleanString('%s_%s_%s' % (anonName, self.getTag('StudyID', ifNotFound='StudyID-unknown'),
                                         self.getTag('StudyDate', ifNotFound='StudyData-unknown')))
        else:
            ss = dcmTools.cleanString('%s_%s_%s' % (self.getTag('PatientName', ifNotFound='NAME-unknown'),
                                            self.getTag('StudyID', ifNotFound='StudyID-unknown'),
                                            self.getTag('StudyDate', ifNotFound='StudyData-unknown')))
        return os.path.join(rootDir, studyPrefix+ss)

class DicomStudy(list):
    """
    Extends list of DicomSeries objects (creating list of list of pydicom.dataset)
    """
    def __init__(self, dSeriesList, OVERVIEW=False, HIDE_PROGRESSBAR=False):
        """
        Set OVERVIEW = False to read pixel data as well (at a cost)
        """
        self.OVERVIEW = OVERVIEW
        self.HIDE_PROGRESSBAR = HIDE_PROGRESSBAR # FIXME - if set need ot pass down to series also - for the writing
        list.__init__(self, dSeriesList)

    @classmethod
    def setFromDictionary(cls, dicomDict, OVERVIEW=False, HIDE_PROGRESSBAR=False, FORCE_READ=False):
        dStudyList = ListOfDicomStudies.setFromDcmDict(dicomDict, OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ)
        if len(dStudyList) > 1:
            raise ValueError('More than one study found - use ListOfDicomStudies class')
        # dicomDirs = getAllDirsUnderRootWithDicoms(dirName) # this was meant to work as a quick shortcut - but potential problems if used incorrectly
        # dSeriesList = []
        # for iDir in dicomDirs:
        #     dSeriesList.append(DicomSeries.setFromDirectory(iDir, OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR))
        return cls(dStudyList[0], OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR)

    @classmethod
    def setFromDirectory(cls, dirName, OVERVIEW=False, HIDE_PROGRESSBAR=False, FORCE_READ=False, ONE_FILE_PER_DIR=False):
        dicomDict = dcmTools.organiseDicomHeirachyByUIDs(dirName, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ, ONE_FILE_PER_DIR=ONE_FILE_PER_DIR, OVERVIEW=OVERVIEW)
        return DicomStudy.setFromDictionary(dicomDict, OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ)


    def __str__(self):
        return '%s: %s: %d series, %d dicoms'%(self.getTag('PatientName'),
                                                self.getTag('SeriesDescription'),
                                                len(self),
                                                self.getNumberOfDicoms())

    def getTag(self, tag, seriesID=0, instanceID=0, ifNotFound='Unknown'):
        return self[seriesID].getTag(tag, dsID=instanceID, ifNotFound=ifNotFound)

    def getTagListAndNames(self, tagList, seriesID=0, instanceID=0):
        return self[seriesID].getTagListAndNames(tagList, dsID=instanceID)

    def isCompressed(self):
        return self[0].isCompressed()

    def getNumberOfDicoms(self):
        return sum([len(i) for i in self])

    def getTopDir(self):
        return os.path.split(os.path.split(self[0][0].filename)[0])[0]

    def getSeriesMatchingDescription(self, matchingStrList, RETURN_SERIES_OBJ=False, REDUCE_MULTIPLE=False):
        matchStrList_lower = [i.lower() for i in matchingStrList]
        possibles = []
        for i in self:
            sDesc = i.getTag('SeriesDescription').lower()
            if any([j in sDesc  for j in matchStrList_lower ]):
                possibles.append(i)
        minID = None
        if len(possibles) == 0:
            return None
        elif (len(possibles) > 1) and REDUCE_MULTIPLE:
            seNums = [int(i.getTag('SeriesNumber')) for i in possibles]
            seNums = [i for i in seNums if i!=0]
            for k1 in range(1, len(seNums)):
                if seNums[k1] < seNums[minID]:
                    minID = k1
        if minID is not None:
            possibles = possibles[minID]
        if RETURN_SERIES_OBJ:
            return possibles
        return [f'SE{possibles[i].getTag("SeriesnNumber")}:{possibles[i].getTag("SeriesDescription")}' for i in range(len(possibles))]

    def getStudyOverview(self, tagList=STUDY_OVERVIEW_TAG_LIST):
        return self.getTagListAndNames(tagList)

    def getPatientOverview(self, tagList=SUBJECT_OVERVIEW_TAG_LIST):
        return self.getTagListAndNames(tagList)

    def getSeriesByID(self, ID):
        for iSeries in self:
            try:
                if int(iSeries.getTag('SeriesNumber')) == ID:
                    return iSeries
            except TypeError: # If SE# == None (report of Sec Capture etc)
                pass

    def getSeriesBySeriesNumber(self, seNum):
        return self.getSeriesByID(seNum)

    def getSeriesByUID(self, UID):
        for iSeries in self:
            if iSeries.getTag('SeriesInstanceUID') == UID:
                return iSeries

    def getStudySummaryDict(self, FORCE_STRING_KEYS=False):
        pt,pv = self.getPatientOverview()
        studySummaryDict = dict(zip(pt, pv))
        st,sv = self.getStudyOverview()
        stdyDict = dict(zip(st, sv))
        studySummaryDict.update(stdyDict)
        listSerDict = []
        for i in self:
            szt, szv = i.getSeriesOverview()
            listSerDict.append(dict(zip(szt, szv)))
        studySummaryDict['Series'] = listSerDict
        return studySummaryDict

    def getStudySummary(self, FULL=True):
        pt,pv = self.getPatientOverview()
        patStr = ','.join([' %s:%s'%(str(i), str(j)) for i,j in zip(pt,pv)])
        st,sv = self.getStudyOverview()
        studyStr = ','.join([' %s:%s'%(str(i), str(j)) for i,j in zip(st,sv)])
        if FULL:
            sst,_ = self[0].getSeriesOverview()
            seriesHeader = ','.join([str(i) for i in sst])
            seriesStr = [','.join([str(i2) for i2 in i.getSeriesOverview()[1]]) for i in self]
            try:
                seriesStr = sorted(seriesStr, key=lambda i:int(i.split(',')[0]))
            except ValueError:
                pass # Try to sort, but if fail (due to return not int somehow, then no sort)
            return 'SUBJECT:%s\nSTUDY:%s\n    %s\n    %s\nTotal: %d images'%(patStr, studyStr, seriesHeader,'\n    '.join(seriesStr), self.getNumberOfDicoms())
        else:
            return 'SUBJECT:%s\nSTUDY:%s'%(patStr, studyStr)

    def getTagValuesList(self, tagList, RETURN_STR):
        output = []
        for i in self:
            output += i.getTagValuesList(tagList, False)
        if RETURN_STR:
            return dcmTools._tagValuesListToString(tagList, output)
        return output

    def writeToOrganisedFileStructure(self, patientOutputDir, anonName=None, anonID='', SE_RENAME={}, studyPrefix=''):
        self.checkIfShouldUse_SAFE_NAMING()
        studyOutputDir = self[0].getStudyOutputDir(patientOutputDir, anonName, studyPrefix)
        UIDupdateDict = {'StudyInstanceUID': dicom.uid.generate_uid()}
        for iSeries in self:
            iSeries.writeToOrganisedFileStructure(studyOutputDir, anonName=anonName, anonID=anonID, UIDupdateDict=UIDupdateDict, SE_RENAME=SE_RENAME, SAFE_NAMING_CHECK=False)
        return studyOutputDir

    def writeToZipArchive(self, patientOutputDir, anonName=None, anonID='', SE_RENAME={}, studyPrefix='', CLEAN_UP=True):
        studyOutputDir = self.writeToOrganisedFileStructure(patientOutputDir, anonName, anonID, SE_RENAME, studyPrefix)
        r, f = os.path.split(studyOutputDir)
        shutil.make_archive(studyOutputDir, 'zip', os.path.join(r, f))
        fileOut = studyOutputDir+'.zip'
        if CLEAN_UP:
            shutil.rmtree(studyOutputDir)
        return fileOut

    def checkIfShouldUse_SAFE_NAMING(self):
        se_instance_set = set()
        for i in self:
            i.checkIfShouldUse_SAFE_NAMING(se_instance_set)


class ListOfDicomStudies(list):
    """
    Extends list of DicomStudies objects (creating list of list of list of pydicom.dataset)
    """
    def __init__(self, dStudiesList, OVERVIEW=False, HIDE_PROGRESSBAR=False):
        """
        Set OVERVIEW = False to read pixel data as well (at a cost)
        """
        self.OVERVIEW = OVERVIEW
        self.HIDE_PROGRESSBAR = HIDE_PROGRESSBAR
        list.__init__(self, dStudiesList)

    @classmethod
    def setFromInput(cls, input, OVERVIEW=False, HIDE_PROGRESSBAR=False, FORCE_READ=False, ONE_FILE_PER_DIR=False):
        if os.path.isdir(input):
            return ListOfDicomStudies.setFromDirectory(input, OVERVIEW=OVERVIEW, 
                                                                HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, 
                                                                FORCE_READ=FORCE_READ, 
                                                                ONE_FILE_PER_DIR=ONE_FILE_PER_DIR)
        if os.path.isfile(input):
            if input.endswith('tar'):
                return ListOfDicomStudies.setFromTar(input, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, 
                                                            FORCE_READ=FORCE_READ)
            elif input.endswith('tar.gz'):
                return ListOfDicomStudies.setFromTar(input, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, 
                                                            FORCE_READ=FORCE_READ)
            elif input.endswith('zip'):
                return ListOfDicomStudies.setFromZip(input, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, 
                                                            FORCE_READ=FORCE_READ)
            else:
                dcmDict = {}
                try:
                    dcmTools.readDicomFile_intoDict(input, dcmDict, FORCE_READ=FORCE_READ, OVERVIEW=OVERVIEW)
                    return ListOfDicomStudies.setFromDcmDict(dcmDict, OVERVIEW, HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ)
                except dicom.filereader.InvalidDicomError:
                    raise IOError("ERROR READING DICOMS: SPDCMTK capable to read dicom files from directory, zip, tar or tar.gz\n")

    @classmethod
    def setFromDirectory(cls, dirName, OVERVIEW=False, HIDE_PROGRESSBAR=False, FORCE_READ=False, ONE_FILE_PER_DIR=False):
        dicomDict = dcmTools.organiseDicomHeirachyByUIDs(dirName, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ, ONE_FILE_PER_DIR=ONE_FILE_PER_DIR, OVERVIEW=OVERVIEW)
        return ListOfDicomStudies.setFromDcmDict(dicomDict, OVERVIEW, HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ)

    @classmethod
    def setFromDcmDict(cls, dicomDict, OVERVIEW=False, HIDE_PROGRESSBAR=False, FORCE_READ=False):
        dStudiesList = []
        for studyUID in dicomDict.keys():
            dSeriesList = []
            for seriesUID in dicomDict[studyUID].keys():
                dSeriesList.append(DicomSeries(dicomDict[studyUID][seriesUID], OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ))
            dStudiesList.append(DicomStudy(dSeriesList, OVERVIEW, HIDE_PROGRESSBAR))
        return cls(dStudiesList, OVERVIEW, HIDE_PROGRESSBAR)

    @classmethod
    def setFromTar(cls, tarFileName, HIDE_PROGRESSBAR=False, FORCE_READ=False, FIRST_ONLY=False, matchTagPair=None):
        # Note need OVERVIEW = False as only get access to file (and pixels) on untaring (maybe only if tar.gz) 
        dicomDict = dcmTools.getDicomDictFromTar(tarFileName, FORCE_READ=FORCE_READ, FIRST_ONLY=FIRST_ONLY, OVERVIEW_ONLY=False,
                                    matchingTagValuePair=matchTagPair, QUIET=True)
        return ListOfDicomStudies.setFromDcmDict(dicomDict, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ)

    @classmethod
    def setFromZip(cls, zipFileName, HIDE_PROGRESSBAR=False, FORCE_READ=False, FIRST_ONLY=False, matchTagPair=None):
        dicomDict = dcmTools.getDicomDictFromZip(zipFileName, FORCE_READ=FORCE_READ, FIRST_ONLY=FIRST_ONLY, OVERVIEW_ONLY=False,
                                    matchingTagValuePair=matchTagPair, QUIET=True)
        return ListOfDicomStudies.setFromDcmDict(dicomDict, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ)


    def __str__(self):
        return '%d studies, %d dicoms'%(len(self), self.getNumberOfDicoms())

    def setSafeNameMode(self):
        for iStudy in self:
            for iSeries in iStudy:
                iSeries.SAFE_NAME_MODE = True

    def isCompressed(self):
        return self[0].isCompressed()

    def getNumberOfDicoms(self):
        return sum([i.getNumberOfDicoms() for i in self])

    def writeToOrganisedFileStructure(self, outputRootDir, anonName=None, anonID='', SE_RENAME={}):
        outDirs = []
        for iStudy in self:
            ooD = iStudy.writeToOrganisedFileStructure(outputRootDir, anonName=anonName, anonID=anonID, SE_RENAME=SE_RENAME)
            outDirs.append(ooD)
        return outDirs

    def writeToZipArchive(self, outputRootDir, anonName=None, anonID='', SE_RENAME={}, CLEAN_UP=True):
        outDirs = []
        for iStudy in self:
            ooD = iStudy.writeToZipArchive(outputRootDir, anonName=anonName, anonID=anonID, SE_RENAME=SE_RENAME, CLEAN_UP=CLEAN_UP)
            outDirs.append(ooD)
        return outDirs

    def getSummaryString(self, FULL=True):
        output = [i.getStudySummary(FULL) for i in self]
        return '\n\n'.join(output)

    def getTableOfTagValues(self, DICOM_TAGS, RETURN_STR):
        output = []
        for i in self:
            output += i.getTagValuesList(DICOM_TAGS, False)
        if RETURN_STR:
            return dcmTools._tagValuesListToString(DICOM_TAGS, output)
        return output

    def getStudyByDate(self, date_str):
        for i in self:
            if i.getTag('StudyDate') == date_str:
                return i

    def buildMSTable(self, DICOM_TAGS=SERIES_OVERVIEW_TAG_LIST):
        pass
        #TODO - need to pass series name or something to query and then calc mean / stdev etc

## =====================================================================================================================
##   BIDS
## =====================================================================================================================
class BIDSConst(object):
    anat = 'anat'
    func = 'func'
    dwi = 'dwi'
    fmap = 'fmap'
    code = 'code'
    configf = 'dcm2bids_config.json'


def getJsonDict(niiORjsonFile):
    if niiORjsonFile.endswith('.nii.gz'):
        niiORjsonFile = niiORjsonFile.replace('.nii.gz','.json')
    with open(niiORjsonFile, 'r') as f:
        jsonDict = json.load(f)
    return jsonDict

def getTagFromJson(niiORjsonFile, tag):
    jsonDict = getJsonDict(niiORjsonFile)
    return jsonDict[tag]


class BIDS_Subject(object):
    """
    Class to work with a single BIDS subject
    """
    def __init__(self, subjID, BIDSDirObj):
        if subjID.startswith('sub-'):
            self.subjID = subjID
        else:
            self.subjID = 'sub-%s'%(subjID)
        self.parent = BIDSDirObj

    def __gt__(self, other):
        return self.subjID > other.subjID

    def __eq__(self, other):
        return self.subjID == other.subjID

    def getSubjID(self):
        return self.subjID.replace('sub-', '')

    def getSessionDir(self, session):
        if not session.startswith('ses-'):
            session = 'ses-%s'%(session)
        return os.path.join(self.parent.rootDir, self.subjID, session)

    def getTypeDir(self, TYPE, session):
        return os.path.join(self.getSessionDir(session), TYPE)

    def _getAllTypeNII(self, TYPE, session):
        dd = self.getTypeDir(TYPE, session)
        return [os.path.join(dd, i) for i in os.listdir(dd) if i.endswith('nii.gz')]

    def getNIIMatchingModality(self, modality, TYPE, session):
        matchingF = [i for i in self._getAllTypeNII(TYPE, session) if i.endswith('_%s.nii.gz'%(modality))]
        return matchingF

    def getNIIMatchingModality_condition(self, modality, TYPE, session, tag, func):
        matchingNII = self.getNIIMatchingModality(modality, TYPE, session)
        if len(matchingNII) == 0:
            return None
        elif len(matchingNII) == 1:
            return matchingNII[0]
        else: # more than 1 run so need to make decision
            tagList = [getTagFromJson(i, tag) for i in matchingNII]
            ID = func(tagList)
            return matchingNII[ID]


class BIDS_Directory(object):
    """
    Class to work with BIDS directory
    """
    def __init__(self, rootDirectroy):
        self.rootDir = rootDirectroy

    def getConfigFile(self, configFName=BIDSConst.configf):
        return os.path.join(self.rootDir, BIDSConst.code, configFName)

    def getSubjList(self):
        subjIDs = [i for i in os.listdir(self.rootDir) if i.startswith('sub-')]
        return sorted([BIDS_Subject(i, self) for i in subjIDs])

def walkdir(folder):
    """Walk through each files in a directory"""
    for dirpath, _, files in os.walk(folder):
        for filename in files:
            yield os.path.abspath(os.path.join(dirpath, filename))


def organiseDicoms(dcmDirectory, outputDirectory, anonName=None, anonID='', FORCE_READ=False, HIDE_PROGRESSBAR=False):
    """ Recurse down directory tree - grab dicoms and move to new
        hierarchical folder structure
    """
    if not HIDE_PROGRESSBAR:
        print('READING...')
    studies = ListOfDicomStudies.setFromInput(dcmDirectory, FORCE_READ=FORCE_READ, OVERVIEW=False, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR)
    if not HIDE_PROGRESSBAR:
        print('WRITTING...')
    res = studies.writeToOrganisedFileStructure(outputDirectory, anonName=anonName, anonID=anonID)
    if res is None:
        print('    ## WARNING - No valid dicoms found')


def studySummary(pathToDicoms):
    """
    :param pathToDicoms: path to dicoms - anywhere in the tree above - will search down tree till find a dicom
    :return: a formatted string
    """
    dStudies = ListOfDicomStudies.setFromDirectory(pathToDicoms)
    return dStudies.getSummaryString()



## =====================================================================================================================
##   WRITE DICOMS
## =====================================================================================================================
def writeVTIToDicoms(vtiFile, dcmTemplateFile_or_ds, outputDir, arrayName=None, tagUpdateDict=None, patientMatrixDict={}):
    if type(vtiFile) == str:
        vti = dcmVTKTK.readVTKFile(vtiFile)
    if arrayName is None:
        A = dcmVTKTK.getScalarsAsNumpy(vti)
    else:
        A = dcmVTKTK.getArrayAsNumpy(vti, arrayName)
    A = np.reshape(A, vti.GetDimensions(), 'F')
    A = np.rot90(A)
    A = np.flipud(A)
    print(A.shape, type(A), A.dtype)
    patientMatrixDict = dcmVTKTK.getPatientMatrixDict(vti, patientMatrixDict)
    return writeNumpyArrayToDicom(A, dcmTemplateFile_or_ds, patientMatrixDict, outputDir, tagUpdateDict=tagUpdateDict)


def writeNumpyArrayToDicom(pixelArray, dcmTemplate_or_ds, patientMatrixDict, outputDir, tagUpdateDict=None):
    """
    patientMatrixDict = {PixelSpacing: [1,2], ImagePositionPatient: [1,3], ImageOrientationPatient: [1,6], SliceThickness: 1}
    Note - use "SliceThickness" in patientMatrixDict - with assumption that SliceThickness==SpacingBetweenSlices (explicitely built this way - ndArray can not have otherwise)
    """
    if tagUpdateDict is None:
        tagUpdateDict = {}
    if dcmTemplate_or_ds is None:
        dsRAW = dcmTools.buildFakeDS()
    elif type(dcmTemplate_or_ds) == str:
        dsRAW = dicom.read_file(dcmTemplate_or_ds)
    else:
        dsRAW = dcmTemplate_or_ds
    nRow, nCol, nSlice = pixelArray.shape
    if pixelArray.dtype != np.int16:
        pixelArray = pixelArray * (2**13 / np.max(pixelArray) )
        pixelArray = pixelArray.astype(np.int16)
    mx, mn = np.max(pixelArray), 0
    try:
        slice0 = tagUpdateDict.pop('SliceLocation0')
    except KeyError:
        slice0 = 0.0

    SeriesUID = dicom.uid.generate_uid()
    try:
        SeriesNumber = tagUpdateDict.pop('SeriesNumber')
    except KeyError:
        try:
            SeriesNumber = dsRAW.SeriesNumber * 100
        except AttributeError:
            SeriesNumber = 99
    dsList = []
    for k in range(nSlice):
        ds = copy.deepcopy(dsRAW)
        ds.SeriesInstanceUID = SeriesUID
        ds.SOPInstanceUID = dicom.uid.generate_uid()
        # ds.SpecificCharacterSet = 'ISO_IR 100'
        # ds.SOPClassUID = 'SecondaryCaptureImageStorage'
        ds.Rows = nRow
        ds.Columns = nCol
        ds.ImagesInAcquisition = nSlice
        ds.InStackPositionNumber = k+1
        # ds.RawDataRunNumber = k+1
        ds.SeriesNumber = SeriesNumber
        ds.InstanceNumber = k+1
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.SliceThickness = patientMatrixDict['SpacingBetweenSlices'] # Can no longer claim overlapping slices if have modified
        ds.SpacingBetweenSlices = patientMatrixDict['SpacingBetweenSlices']
        ds.SmallestImagePixelValue = int(mn)
        ds.LargestImagePixelValue = int(mx)
        ds.WindowCenter = int(mx / 2)
        ds.WindowWidth = int(mx / 2)
        ds.PixelSpacing = list(patientMatrixDict['PixelSpacing'])
        kVec = np.cross(patientMatrixDict['ImageOrientationPatient'][:3],
                        patientMatrixDict['ImageOrientationPatient'][3:])
        ImagePositionPatient = np.array(patientMatrixDict['ImagePositionPatient']) + k*kVec*ds.SpacingBetweenSlices
        ds.ImagePositionPatient = list(ImagePositionPatient)
        # try:
        sliceLoc = slice0 + k*ds.SpacingBetweenSlices
        # except KeyError:
        #     sliceLoc = dcmTools.distPts(ImagePositionPatient, np.array(patientMatrixDict['ImagePositionPatient']))
        ds.SliceLocation = sliceLoc
        ds.ImageOrientationPatient = list(patientMatrixDict['ImageOrientationPatient'])
        for iKey in tagUpdateDict.keys():
            try:
                ds[iKey] = tagUpdateDict[iKey]
            except (ValueError, TypeError):
                ds.add_new(tagUpdateDict[iKey][0], tagUpdateDict[iKey][1], tagUpdateDict[iKey][2])
        ds.PixelData = pixelArray[:,:,k].tostring()
        ds['PixelData'].VR = 'OW'
        dsList.append(ds)
    dcmSeries = DicomSeries(dsList, HIDE_PROGRESSBAR=True)
    dcmSeries.writeToOrganisedFileStructure(outputDir)



def getResolution(dataVts):
    o,p1,p2,p3 = [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]
    i0,i1,j0,j1,k0,k1 = dataVts.GetExtent()
    print(i0,i1,j0,j1,k0,k1)
    dataVts.GetPoint(i0,j0,k0, o)
    dataVts.GetPoint(i0+1,j0,k0, p1)
    dataVts.GetPoint(i0,j0+1,k0, p2)
    dataVts.GetPoint(i0,j0,k0+1, p3)
    di = abs(distBetweenTwoPts(p1, o))
    dj = abs(distBetweenTwoPts(p2, o))
    dk = abs(distBetweenTwoPts(p3, o))
    return [di, dj, dk]

def distBetweenTwoPts(a, b):
    return np.sqrt(np.sum((np.asarray(a) - np.asarray(b)) ** 2))

def printPoints(vtsD):
    dd = vtsD.GetDimensions()
    ijks = [[0,0,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4]]
    for ii in ijks:
        ID = np.ravel_multi_index(ii, dd, order='F')
        print(ii, ID, vtsD.GetPoint(ID))