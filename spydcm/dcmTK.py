# -*- coding: utf-8 -*-

"""Classes for working with Dicom studies
"""

import os
import pydicom as dicom
from tqdm import tqdm
import glob
import json
import numpy as np

# Local imports 
import spydcm.dcmUtils as dcmUtils
from defaults import DEFAULT_SERIES_OVERVIEW_TAG_LIST, DEFAULT_STUDY_OVERVIEW_TAG_LIST, DEFAULT_SUBJECT_OVERVIEW_TAG_LIST
import spydcm.dcmTools as dcmTools




## =====================================================================================================================
##        CLASSES
## =====================================================================================================================
class DicomSeries(list):
    """
    Extends a list of ds (pydicom dataset) objects.
    """
    def __init__(self, dsList=None, OVERVIEW=True, HIDE_PROGRESSBAR=False):
        """
        Set OVERVIEW = False to read pixel data as well (at a cost)
        """
        if dsList is None:
            dsList = []
        self.OVERVIEW = OVERVIEW
        self.HIDE_PROGRESSBAR = HIDE_PROGRESSBAR
        list.__init__(self, dsList)

    def __str__(self):
        return ' '.join([str(i) for i in self.getSeriesOverview()[1]])

    @classmethod
    def setFromDirectory(cls, dirName, OVERVIEW=True, HIDE_PROGRESSBAR=False, FORCE_READ=False, ONE_FILE_PER_DIR=False):
        dicomDict = dcmUtils.organiseDicomHeirachyByUIDs(dirName, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ, ONE_FILE_PER_DIR=ONE_FILE_PER_DIR, OVERVIEW=OVERVIEW)
        dStudyList = ListOfDicomStudies.setFromDcmDict(dicomDict, OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE=FORCE_READ)
        if len(dStudyList) > 1:
            raise ValueError('More than one study found - use ListOfDicomStudies class')
        dStudy = dStudyList[0]
        if len(dStudy) > 1:
            raise ValueError('More than one series found - use DicomStudy class')
        return cls(dStudy[0], OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR)

    @classmethod
    def setFromFileList(cls, fileList, OVERVIEW=True, HIDE_PROGRESSBAR=False, FORCE=False):
        dsList = [dicom.read_file(i, stop_before_pixels=OVERVIEW, force=FORCE) for i in fileList]
        return cls(dsList, OVERVIEW, HIDE_PROGRESSBAR)

    def getRootDir(self):
        return os.path.split(self[0].filename)[0]

    def sortByInstanceNumber(self):
        self.sort(key=dcmTools.instanceNumberSortKey)

    def getTag(self, tag, dsID=0, ifNotFound='Unknown'):
        try:
            tt = self.getTagObj(tag, dsID)
            return tt.value
        except KeyError:
            return ifNotFound

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

    def getSeriesOverview(self, tagList=DEFAULT_SERIES_OVERVIEW_TAG_LIST):
        names, vals = self.getTagListAndNames(tagList)
        names.append('ImagesInSeries')
        vals.append(len(self))
        return names, vals

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

    def writeToOrganisedFileStructure(self, studyOutputDir, anonName=None, FORCED_READ=False, SE_RENAME={}):
        """ Recurse down directory tree - grab dicoms and move to new
            hierarchical folder structure
            SE_RENAME = dict of SE# and Name to rename the SE folder    
        """
        ADD_TRANSFERSYNTAX = False
        LIKE_ORIG = True
        destFile = None
        seriesOutDirName = self.getSeriesOutDirName(SE_RENAME)
        seriesOutputDir = os.path.join(studyOutputDir, seriesOutDirName)
        if FORCED_READ:
            ADD_TRANSFERSYNTAX = True ## THIS IS A BESPOKE CHANGE
        TO_DECOMPRESS = self.isCompressed()
        for ds in tqdm(self.yieldDataset(), total=len(self), disable=self.HIDE_PROGRESSBAR):
            if TO_DECOMPRESS:
                ds.decompress('gdcm')
            if ADD_TRANSFERSYNTAX:
                ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
                LIKE_ORIG=False
            destFile = dcmUtils.writeOut_ds(ds, seriesOutputDir, anonName, WRITE_LIKE_ORIG=LIKE_ORIG)
        return destFile

    def writeToNII(self, outputPath, outputNaming=('PatientName', 'SeriesNumber', 'SeriesDescription')):
        dcm2niiCmd = "dcm2nii -p n -e y -d n -x n -o '%s' '%s'"%(outputPath, self.getRootDir())
        print('RUNNING: %s'%(dcm2niiCmd))
        os.system(dcm2niiCmd)
        list_of_files = glob.glob(os.path.join(outputPath, '*.nii.gz'))  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        fileName = '_'.join([str(self.getTag(i)) for i in outputNaming])
        fileName = dcmTools.cleanString(fileName)
        newFileName = os.path.join(outputPath, '%s.nii.gz'%(fileName))
        os.rename(latest_file, newFileName)
        print('Made %s --> as %s'%(latest_file, newFileName))

    def writeToPVD(self, outputPath, outputNaming=('PatientName', 'SeriesNumber', 'SeriesDescription')):
        pass        

    def getSliceLocations(self):
        return [self.getTag('SliceLocation', i) for i in range(len(self))]

    def getNumberOfSlicesPerVolume(self):
        sliceLoc = self.getSliceLocations()
        return len(set(sliceLoc))

    def getNumberOfTimeSteps(self):
        sliceLoc = self.getSliceLocations()
        sliceLocS = set(sliceLoc)
        return sliceLoc.count(sliceLocS.pop())

    def getPixelDataAsNumpy(self):
        """Get pixel data as numpy array organised by slice and time(if present).
            Also return dictionary of meta ('spacing', 'origin', 'imagepositionpatient cosines')
        """
        I,J,K = int(self.getTag('Columns')), int(self.getTag('Rows')), int(self.getNumberOfSlicesPerVolume())
        N = self.getNumberOfTimeSteps()
        A = np.zeros((I, J, K, N))
        

    def getScanDuration_secs(self):
        try:
            return self.getTag(0x0019105a, ifNotFound=0.0) / 1000000.0
        except AttributeError:
            return 0.0

    def getDeltaRow(self):
        return float(self.getTag('PixelSpacing', ifNotFound=[0.0,0.0])[0])

    def getDeltaCol(self):
        return float(self.getTag('PixelSpacing', ifNotFound=[0.0,0.0])[1])

    def getTemporalResolution(self):
        try:
            return float(self.getTag('NominalInterval', ifNotFound=0.0)/self.getTag('CardiacNumberOfImages', ifNotFound=1))
        except ZeroDivisionError:
            return 0       

    def getPulseSequenceName(self):
        try:
            return self.getTag(0x0019109c)
        except AttributeError:
            return ''

    def getInternalPulseSequenceName(self):
        try:
            return self.getTag(0x0019109e)
        except AttributeError:
            return ''

    def getSeriesDescription(self):
        return self.getTag('SeriesDescription')

    def getSeriesInfoDict(self):
        outDict = {'SeriesNumber':self.getTag('SeriesNumber'),
            'SeriesDescription':self.getTag('SeriesDescription'),
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
            'InternalPulseSequenceName':self.getInternalPulseSequenceName()}
        outDict['nSlice'] = len(self)
        return outDict



class DicomStudy(list):
    """
    Extends list of DicomSeries objects (creating list of list of pydicom.dataset)
    """
    def __init__(self, dSeriesList, OVERVIEW=True, HIDE_PROGRESSBAR=False):
        """
        Set OVERVIEW = False to read pixel data as well (at a cost)
        """
        self.OVERVIEW = OVERVIEW
        self.HIDE_PROGRESSBAR = HIDE_PROGRESSBAR # FIXME - if set need ot pass down to series also - for the writing
        list.__init__(self, dSeriesList)

    @classmethod
    def setFromDictionary(cls, dicomDict, OVERVIEW=True, HIDE_PROGRESSBAR=False, FORCE_READ=False):
        dStudyList = ListOfDicomStudies.setFromDcmDict(dicomDict, OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE=FORCE_READ)
        if len(dStudyList) > 1:
            raise ValueError('More than one study found - use ListOfDicomStudies class')
        # dicomDirs = getAllDirsUnderRootWithDicoms(dirName) # this was meant to work as a quick shortcut - but potential problems if used incorrectly
        # dSeriesList = []
        # for iDir in dicomDirs:
        #     dSeriesList.append(DicomSeries.setFromDirectory(iDir, OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR))
        return cls(dStudyList[0], OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR)

    @classmethod
    def setFromDirectory(cls, dirName, OVERVIEW=True, HIDE_PROGRESSBAR=False, FORCE_READ=False, ONE_FILE_PER_DIR=False):
        dicomDict = dcmUtils.organiseDicomHeirachyByUIDs(dirName, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ, ONE_FILE_PER_DIR=ONE_FILE_PER_DIR, OVERVIEW=OVERVIEW)
        return DicomStudy.setFromDictionary(dicomDict, OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR)


    def __str__(self):
        return '%s: %s: %d series, %d dicoms'%(self.getTag('PatientName'),
                                                self.getTag('SeriesDescription'),
                                                len(self),
                                                self.getNumberOfDicoms())

    def getTag(self, tag, seriesID=0, instanceID=0, ifNotFound='Unknown'):
        return self[seriesID].getTag(tag, dsID=instanceID, ifNotFound=ifNotFound)

    def getTagListAndNames(self, tagList, seriesID=0, instanceID=0):
        return self[seriesID].getTagListAndNames(tagList, dsID=instanceID)
        # names, vals = [], []
        # for i in tagList:
        #     try:
        #         dataEle = self[seriesID].getTagObj(i, dsID=instanceID)
        #         names.append(str(dataEle.name).replace('[','').replace(']',''))
        #         vals.append(str(dataEle.value))
        #     except KeyError:
        #         names.append(str(i))
        #         vals.append('Unknown')
        # return names, vals

    def isCompressed(self):
        return self[0].isCompressed()

    def getNumberOfDicoms(self):
        return sum([len(i) for i in self])

    def getTopDir(self):
        return os.path.split(os.path.split(self[0][0].filename)[0])[0]

    def getSeriesName(self, matchingStrList, RETURN_SERIES_OBJ=False):
        matchStrList_lower = [i.lower() for i in matchingStrList]
        possibles = []
        for i in self:
            sDesc = i.getTag('SeriesDescription').lower()
            if any([j in sDesc  for j in matchStrList_lower ]):
                possibles.append(i)
        minID = 0
        if len(possibles) == 0:
            return None
        elif len(possibles) > 1:
            seNums = [int(i.getTag('SeriesNumber')) for i in possibles]
            seNums = [i for i in seNums if i!=0]
            for k1 in range(1, len(seNums)):
                if seNums[k1] < seNums[minID]:
                    minID = k1
        if RETURN_SERIES_OBJ:
            return possibles[minID]
        return 'SE%d_%s' % (int(possibles[minID].getTag('SeriesNumber')), possibles[minID].getTag('SeriesDescription'))

    def getStudyOverview(self, tagList=DEFAULT_STUDY_OVERVIEW_TAG_LIST):
        return self.getTagListAndNames(tagList)

    def getPatientOverview(self, tagList=DEFAULT_SUBJECT_OVERVIEW_TAG_LIST):
        return self.getTagListAndNames(tagList)

    def getSeriesByID(self, ID):
        for iSeries in self:
            if int(iSeries.getTag('SeriesNumber')) == ID:
                return iSeries

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

    def __getStudyOutputDir(self, anonName=None):
        if anonName is not None:
            return dcmTools.cleanString('%s_%s' % (self.getTag('StudyID', ifNotFound='StudyID-unknown'),
                                         self.getTag('StudyDate', ifNotFound='StudyData-unknown')))
        else:
            return dcmTools.cleanString('%s_%s_%s' % (self.getTag('PatientName', ifNotFound='NAME-unknown'),
                                            self.getTag('StudyID', ifNotFound='StudyID-unknown'),
                                            self.getTag('StudyDate', ifNotFound='StudyData-unknown')))

    def writeToOrganisedFileStructure(self, patientOutputDir, anonName=None, FORCED_READ=False, SE_RENAME={}):
        studyOutputDir = os.path.join(patientOutputDir, self.__getStudyOutputDir(anonName))
        for iSeries in self:
            iSeries.writeToOrganisedFileStructure(studyOutputDir, anonName=anonName, FORCED_READ=FORCED_READ, SE_RENAME=SE_RENAME)
        return studyOutputDir

class ListOfDicomStudies(list):
    """
    Extends list of DicomStudies objects (creating list of list of list of pydicom.dataset)
    """
    def __init__(self, dStudiesList, OVERVIEW=True, HIDE_PROGRESSBAR=False):
        """
        Set OVERVIEW = False to read pixel data as well (at a cost)
        """
        self.OVERVIEW = OVERVIEW
        self.HIDE_PROGRESSBAR = HIDE_PROGRESSBAR
        list.__init__(self, dStudiesList)

    @classmethod
    def setFromDirectory(cls, dirName, OVERVIEW=True, HIDE_PROGRESSBAR=False, FORCE=False, ONE_FILE_PER_DIR=False):
        dicomDict = dcmUtils.organiseDicomHeirachyByUIDs(dirName, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE, ONE_FILE_PER_DIR=ONE_FILE_PER_DIR, OVERVIEW=OVERVIEW)
        return ListOfDicomStudies.setFromDcmDict(dicomDict, OVERVIEW, HIDE_PROGRESSBAR, FORCE=FORCE)

    @classmethod
    def setFromDcmDict(cls, dicomDict, OVERVIEW=True, HIDE_PROGRESSBAR=False, FORCE=False):
        dStudiesList = []
        for studyUID in dicomDict.keys():
            dSeriesList = []
            for seriesUID in dicomDict[studyUID].keys():
                if dicomDict[studyUID][seriesUID][0].filename.endswith('.tar') or dicomDict[studyUID][seriesUID][0].filename.endswith('.tar.gz'):
                    dSeriesList.append(DicomSeries(dicomDict[studyUID][seriesUID], OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR))
                else:
                    fileList = [i.filename for i in dicomDict[studyUID][seriesUID]]
                    dSeriesList.append(DicomSeries.setFromFileList(fileList, OVERVIEW, HIDE_PROGRESSBAR, FORCE=FORCE))
            dStudiesList.append(DicomStudy(dSeriesList, OVERVIEW, HIDE_PROGRESSBAR))
        return cls(dStudiesList, OVERVIEW, HIDE_PROGRESSBAR)

    @classmethod
    def setFromTar(cls, tarFileName, OVERVIEW=False, HIDE_PROGRESSBAR=False, FORCE_READ=False, FIRST_ONLY=False, matchTagPair=None):
        # Note need OVERVIEW = False as only get access to file (and pixels) on untaring (maybe only if tar.gz) 
        dicomDict = dcmUtils.getDicomDictFromTar(tarFileName, FORCE_READ=FORCE_READ, FIRST_ONLY=FIRST_ONLY, OVERVIEW_ONLY=OVERVIEW,
                                    matchingTagValuePair=matchTagPair, QUIET=True)
        return ListOfDicomStudies.setFromDcmDict(dicomDict, OVERVIEW, HIDE_PROGRESSBAR, FORCE=FORCE_READ)


    def __str__(self):
        return '%d studies, %d dicoms'%(len(self), self.getNumberOfDicoms())

    def isCompressed(self):
        return self[0].isCompressed()

    def getNumberOfDicoms(self):
        return sum([i.getNumberOfDicoms() for i in self])

    def writeToOrganisedFileStructure(self, outputRootDir, anonName=None, FORCED_READ=False, SE_RENAME={}):
        outDirs = []
        for iStudy in self:
            ooD = iStudy.writeToOrganisedFileStructure(outputRootDir, anonName=anonName, FORCED_READ=FORCED_READ, SE_RENAME=SE_RENAME)
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

    def buildMSTable(self, DICOM_TAGS=DEFAULT_SERIES_OVERVIEW_TAG_LIST):
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



def organiseDicoms(dcmDirectory, outputDirectory, anonName=None, FORCE_READ=False, HIDE_PROGRESSBAR=False):
    """ Recurse down directory tree - grab dicoms and move to new
        hierarchical folder structure
    """
    if not HIDE_PROGRESSBAR:
        print('READING...')
    if os.path.isfile(dcmDirectory):
        studies = ListOfDicomStudies.setFromTar(dcmDirectory, FORCE_READ=FORCE_READ, OVERVIEW=False, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR)
    else:      
       studies = ListOfDicomStudies.setFromDirectory(dcmDirectory, FORCE=FORCE_READ, OVERVIEW=False, ONE_FILE_PER_DIR=False, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR)
    if not HIDE_PROGRESSBAR:
        print('WRITTING...')
    res = studies.writeToOrganisedFileStructure(outputDirectory, anonName=anonName, FORCED_READ=FORCE_READ)
    if res is None:
        print('    ## WARNING - No valid dicoms found')


def studySummary(pathToDicoms):
    """
    :param pathToDicoms: path to dicoms - anywhere in the tree above - will search down tree till find a dicom
    :return: a formatted string
    """
    print('READING...')
    dStudies = ListOfDicomStudies.setFromDirectory(pathToDicoms)
    return dStudies.getSummaryString()


