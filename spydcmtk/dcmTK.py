# -*- coding: utf-8 -*-

"""Classes for working with Dicom studies
"""
import os
import pydicom as dicom
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from datetime import datetime
from tqdm import tqdm
import json
from multiprocessing import Pool
import numpy as np
import shutil
import matplotlib.pyplot as plt
# Local imports 
import spydcmtk.dcmTools as dcmTools
import spydcmtk.dcmVTKTK as dcmVTKTK
from spydcmtk.spydcm_config import SpydcmTK_config




## =====================================================================================================================
##        CLASSES
## =====================================================================================================================
class DicomSeries(list):
    """Extends a list of ds (pydicom dataset) objects.
    """
    def __init__(self, dsList=None, OVERVIEW=False, HIDE_PROGRESSBAR=False, FORCE_READ=False, SAFE_NAME_MODE=False):
        """Initialise DicomSeries with list of pydicom datasets.

        Args:
            dsList (list, optional): list of pydicom dataset. Defaults to None.
            OVERVIEW (bool, optional): Set True to not read pixel data, for efficiency in applications of meta 
                                        data only interest. 
                                        Pass to pydicom.filereader.dcmread parameter:stop_before_pixels
                                        Defaults to False.
            HIDE_PROGRESSBAR (bool, optional): Set True to hide tqdm progressbar. Defaults to False.
            FORCE_READ (bool, optional): Force read dicom files even if they do not conform to DICOM standard. 
                                            Pass to pydicom.filereader.dcmread parameter:force
                                            Defaults to False.
            SAFE_NAME_MODE (bool, optional): Set writing mode to use UIDs for file naming. Defaults to False.
        """
        if dsList is None:
            dsList = []
        self.OVERVIEW = OVERVIEW
        self.HIDE_PROGRESSBAR = HIDE_PROGRESSBAR
        self.FORCE_READ = FORCE_READ
        self.SAFE_NAME_MODE = SAFE_NAME_MODE
        self.NOT_FULLY_LOADED = False
        list.__init__(self, dsList)

    def __str__(self):
        """Return output of getSeriesOverview method
        """
        return ' '.join([str(i) for i in self.getSeriesOverview()[1]])


    def __getitem__(self, item):
        if isinstance(item, slice):
            result = super().__getitem__(item)
            return DicomSeries(result)
        else:
            return super().__getitem__(item)

    @classmethod
    def _setFromDictionary(cls, dicomDict, OVERVIEW=False, HIDE_PROGRESSBAR=False, FORCE_READ=False):
        dStudyList = ListOfDicomStudies.setFromDcmDict(dicomDict, OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ)
        if len(dStudyList) > 1:
            raise ValueError('More than one study found - use ListOfDicomStudies class')
        try:
            dStudy = dStudyList[0]
        except IndexError:
            raise ValueError('No DICOMS found - please check your inputs')
        if len(dStudy) > 1:
            raise ValueError('More than one series found - use DicomStudy class')
        return cls(dStudy[0], OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ)

    @classmethod
    def setFromDirectory(cls, dirName, OVERVIEW=False, HIDE_PROGRESSBAR=False, FORCE_READ=False, ONE_FILE_PER_DIR=False):
        """Initialise object from directory of dicom files.

        Args:
            dirName (str): Path to directory of dicom files
            OVERVIEW (bool, optional): Set attribute OVERVIEW. Defaults to False.
            HIDE_PROGRESSBAR (bool, optional): Set attribute HIDE_PROGRESSBAR. Defaults to False.
            FORCE_READ (bool, optional): Set attribute FORCE_READ. Defaults to False.
            ONE_FILE_PER_DIR (bool, optional): Read only one file per directory for fast applications
                                                with meta interest only. Defaults to False.

        Raises:
            ValueError: If dicom files belonging to more than one study are found.
            ValueError: If dicom files belonging to more than one series are found.

        Returns:
            DicomSeries: An instance of DicomSeries class.
        """
        dicomDict = dcmTools.organiseDicomHeirarchyByUIDs(dirName, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ, ONE_FILE_PER_DIR=ONE_FILE_PER_DIR, OVERVIEW=OVERVIEW)
        obj = DicomSeries._setFromDictionary(dicomDict, OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ)
        obj.NOT_FULLY_LOADED = ONE_FILE_PER_DIR
        return obj

    @classmethod
    def setFromFileList(cls, fileList, OVERVIEW=False, HIDE_PROGRESSBAR=False, FORCE_READ=False):
        """Initialise object from list of files

        Args:
            fileList (list): List of full file names
            OVERVIEW (bool, optional): Set attribute OVERVIEW. Defaults to False.
            HIDE_PROGRESSBAR (bool, optional): Set attribute HIDE_PROGRESSBAR. Defaults to False.
            FORCE_READ (bool, optional): Set attribute FORCE_READ. Defaults to False.

        Raises:
            ValueError: If dicom files belonging to more than one study are found.
            ValueError: If dicom files belonging to more than one series are found.

        Returns:
            DicomSeries: An instance of DicomSeries class.
        """
        dicomDict = {}
        for iFile in fileList:
            dcmTools.readDicomFile_intoDict(iFile, dicomDict, FORCE_READ=FORCE_READ, OVERVIEW=OVERVIEW)
        return DicomSeries._setFromDictionary(dicomDict, OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ)


    def getRootDir(self):
        """Return the root directory of the series: os.path.split(self[0].filename)[0]
        """
        return os.path.split(self[0].filename)[0]


    def filterByTag(self, tagName, tagValue):
        """Filter the series by a tag value.

        Args:
            tagName (str): The name of the tag to filter by.
            tagValue (str): The value of the tag to filter by.

        Returns:
            DicomSeries: A new DicomSeries instance with the filtered series.
        """
        return DicomSeries([i for i in self if i.getTag(tagName) == tagValue], 
                           OVERVIEW=self.OVERVIEW, 
                           HIDE_PROGRESSBAR=self.HIDE_PROGRESSBAR, 
                           FORCE_READ=self.FORCE_READ)


    def getDicomFullFileName(self, dsID=0):
        """Return the full file name of the dicom file.

        Args:
            dsID (int, optional): The index of the dicom file to return. Defaults to 0.

        Returns:
            str: The full file name of the dicom file.
        """
        return self[dsID].filename


    def sortByInstanceNumber(self):
        """Sort the series by instance number.
        """
        self.sort(key=dcmTools.instanceNumberSortKey)


    def sortBySlice_InstanceNumber(self):
        """Sort the series by slice location and instance number.
        """
        self.sort(key=dcmTools.sliceLoc_InstanceNumberSortKey)


    def getTag(self, tag, dsID=0, ifNotFound='Unknown', convertToType=None):
        """Get the value of a tag.

        Args:
            tag (str): The name of the tag to get.
            dsID (int, optional): The index of the dicom file to get the tag from. Defaults to 0.
            ifNotFound (str, optional): The value to return if the tag is not found. Defaults to 'Unknown'.
            convertToType (function, optional): The function to convert the tag value to. Defaults to None.

        Returns:
            The value of the tag.
        """
        try:
            tt = self.getTagObj(tag, dsID)
            if convertToType is not None:
                return convertToType(tt.value)
            return tt.value
        except KeyError:
            return ifNotFound


    def setTags_all(self, tag, value):
        """Set the value of a tag for all dicom files in the series.

        Args:
            tag (str): The name of the tag to set.
            value (str): The value to set the tag to.
        """
        for ds in self:
            ds[tag].value = value

    def getTagObj(self, tag, dsID=0):
        """Get the tag object for a given tag and dicom file index.

        Args:
            tag (str): The name of the tag to get.
            dsID (int, optional): The index of the dicom file to get the tag from. Defaults to 0.

        Returns:
            The tag object.
        """
        try:
            if str(tag)[:2] == '0x':
                tag = int(tag, 16)
        except TypeError:
            pass # Not string
        return self[dsID][tag]


    def getTagValuesList(self, tagList, RETURN_STR):
        """
        Build table with file name and value from each tag in list.
        
        Args:
            tagList (list): The list of tags to get the values from.
            RETURN_STR (bool): Whether to return the values as a string.

        Returns:
            The list of values.
        """
        valueList = []
        for dsID, i in enumerate(self):
            subList = ['"%s"'%(str(i.filename))] + [str(self.getTag(t, dsID=dsID)) for t in tagList]
            valueList.append(subList)
        if RETURN_STR:
            return dcmTools._tagValuesListToString(tagList, valueList)
        return valueList


    def getTagListAndNames(self, tagList, dsID=0):
        """Get the list of tag names and values.

        Args:
            tagList (list): The list of tags to get the values from.
            dsID (int, optional): The index of the dicom file to get the tag from. Defaults to 0.

        Returns:
            The list of tag names and values.
        """
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


    def tagsToDictionary(self):
        dict_out = {}
        possible_pix_list = ['7FE00010', '7FE00008', '7FE00009', '60xx3000', '00880200'] #, '00287FE0']
        for ds in self:
            dd = ds.to_json_dict()
            for itag in possible_pix_list:
                try:
                    dd.pop(itag) # Remove actual PixelData
                except KeyError:
                    pass # likely hit this multiple (or even all) 
            dict_out[ds.InstanceNumber] = dd
        return dict_out
    

    def tagsToJson(self, jsonFileOut):
        """Convert the series to a json file (minus the pixel data).

        Args:
            jsonFileOut (str): The file to save the json to.

        Returns:
            The file to save the json to.
        """
        dict_out = self.tagsToDictionary()
        dcmTools.writeDictionaryToJSON(jsonFileOut, dict_out)
        return jsonFileOut


    def _loadToMemory(self):
        """Load the series into memory. An internal function to load full data to memory when needed if not already loaded.
        """
        if self.NOT_FULLY_LOADED:
            rootDir = self.getRootDir()
            self.clear()
            self.extend([dicom.dcmread(os.path.join(rootDir, i), force=self.FORCE_READ) for i in tqdm(os.listdir(rootDir), disable=self.HIDE_PROGRESSBAR)])
            self.NOT_FULLY_LOADED = False


    def getSeriesOverview(self, tagList=SpydcmTK_config.SERIES_OVERVIEW_TAG_LIST):
        """Get the series overview as a tuple of two lists (names, values).

        Args:
            tagList (list, optional): The list of tags to get the values from. Defaults to SpydcmTK_config.SERIES_OVERVIEW_TAG_LIST. See spydcmtk.conf

        Returns:
            tuple: A tuple of two lists: the first list contains the names of the tags, and the second list contains the values of the tags.
        """
        names, vals = self.getTagListAndNames(tagList)
        names.append('ImagesInSeries')
        if len(self) == 1: # ONLY READ ONE FILE PER DIR
            vals.append(dcmTools.countFilesInDir(self.getRootDir()))
        else:
            vals.append(len(self))
        return names, vals


    def getSeriesTimeAsDatetime(self):
        """Get the series time as a datetime object.

        Returns:
            datetime: The series time as a datetime object.
        """
        dos = self.getTag('SeriesDate', ifNotFound="19000101")
        tos = self.getTag('SeriesTime', ifNotFound="000000")
        try:
            return datetime.strptime(f"{dos} {tos}", "%Y%m%d %H%M%S.%f")
        except ValueError:
            return datetime.strptime(f"{dos} {tos}", "%Y%m%d %H%M%S")


    def getImagePositionPatient_np(self, dsID):
        """Get the image position patient as a numpy array.
        For 3D DICOM files, checks SharedFunctionalGroupsSequence if standard tags not available.

        Args:
            dsID (int, optional): The index of the dicom file to get the image position patient from. Defaults to 0.

        Returns:
            numpy.ndarray: The image position patient as a numpy array shape (3,).
        """
        if self.has3DPixelData():
            # For 3D DICOM files, try standard tags first, then SharedFunctionalGroupsSequence
            ipp = self.getTag('ImagePositionPatient', dsID=dsID, ifNotFound=None)
            if ipp is not None:
                return np.array(ipp)
            
            # Try to get from SharedFunctionalGroupsSequence
            ipp = self._getImagePositionPatientFromSharedFunctionalGroups()
            if ipp is not None:
                return np.array(ipp)
        
        # Fallback to standard method or default
        ipp = self.getTag('ImagePositionPatient', dsID=dsID, ifNotFound=[0.0,0.0,0.0])
        return np.array(ipp)


    def getImageOrientationPatient_np(self, dsID=0):
        """Get the image orientation patient as a numpy array.
        For 3D DICOM files, checks SharedFunctionalGroupsSequence if standard tags not available.

        Args:
            dsID (int, optional): The index of the dicom file to get the image orientation patient from. Defaults to 0.

        Returns:
            numpy.ndarray: The image orientation patient as a numpy array shape (6,).
        """
        if self.has3DPixelData():
            # For 3D DICOM files, try standard tags first, then SharedFunctionalGroupsSequence
            iop = self.getTag('ImageOrientationPatient', dsID=dsID, ifNotFound=None)
            if iop is not None:
                return np.array(iop)
            
            # Try to get from SharedFunctionalGroupsSequence
            iop = self._getImageOrientationPatientFromSharedFunctionalGroups()
            if iop is not None:
                return np.array(iop)
        
        # Fallback to standard method or default
        iop = self.getTag('ImageOrientationPatient', dsID=dsID, ifNotFound=[1.0,0.0,0.0,0.0,1.0,0.0])
        return np.array(iop)


    def yieldDataset(self):
        for ds in self:
            if self.OVERVIEW:
                yield  dicom.dcmread(ds.filename, stop_before_pixels=False, force=True)
            else:
                yield ds


    def isCompressed(self):
        """Check if the series is compressed.

        Returns:
            bool: True if the series is compressed, False otherwise.
        """
        return dcmTools._isCompressed(self[0])


    def getSeriesNumber(self):
        """Get the series number.

        Returns:
            int: The series number.
        """
        return int(self.getTag('SeriesNumber', ifNotFound=0))


    def getSeriesOutDirName(self, SE_RENAME={}):
        """Build an auto-generated series output directory name. Used for writing to organised file structure.
        If SAFE_NAME_MODE is True, then the series instance UID is used as part of the directory name.
        The entries SpydcmTK_config.SERIES_NAMING_TAG_LIST in spydcmtk.conf are used to build the directory name.

        Args:
            SE_RENAME (dict, optional): A dictionary of series numbers to rename. Defaults to {}.

        Returns:
            str: The series output directory name.
        """
        thisSeNum = self.getTag('SeriesNumber', ifNotFound='#')
        suffix = ''
        if (thisSeNum=="#") or self.SAFE_NAME_MODE:
            suffix += '_'+self.getTag("SeriesInstanceUID")
        for iTag in SpydcmTK_config.SERIES_NAMING_TAG_LIST:
            iVal = self.getTag(iTag, ifNotFound='')
            if len(iVal) > 0:
                suffix += '_'+iVal
        if thisSeNum in SE_RENAME.keys():
            return SE_RENAME[thisSeNum]
        return dcmTools.cleanString(f"SE{thisSeNum}{suffix}")


    # ----------------------------------------------------------------------------------------------------
    def removeTimes(self, timesIDs_ToRemove):
        """Remove times from the series.

        Args:
            timesIDs_ToRemove (list): The list of times to remove.
        """
        K = int(self.getNumberOfSlicesPerVolume())
        self.sortBySlice_InstanceNumber()
        N = self.getNumberOfTimeSteps()
        c0 = 0
        dsToRm = []
        for k1 in range(K):
            for k2 in range(N):
                if k2 in timesIDs_ToRemove:
                    dsToRm.append(self[c0])
                c0 += 1
        self._rm_via_DS_List(dsToRm)


    def removeInstances(self, instanceIDs_to_remove):
        """Remove instances from the series.

        Args:
            instanceIDs_to_remove (list): The list of instance IDs to remove.
        """
        dsToRm = []
        for k1 in range(len(self)):
            if self.getTag("InstanceNumber", k1, convertToType=int) in instanceIDs_to_remove:
                dsToRm.append(self[k1])
        self._rm_via_DS_List(dsToRm)


    def _rm_via_DS_List(self, dsList):
        for iDS in dsList:
            self.remove(iDS)


    def deleteTag(self, tag, dsID):
        """Delete a tag from an entry in the series.

        Args:
            tag (str): The tag to delete.
            dsID (int): The index of the dicom file to delete the tag from.
        """
        tagObj = self.getTagObj(tag, dsID)
        del tagObj


    def deleteTags_fromAll(self, tag):
        """Delete a tag from all entries in the series.

        Args:
            tag (str): The tag to delete.
        """
        for k1 in range(len(self)):
            self.deleteTag(tag, k1)


    # ----------------------------------------------------------------------------------------------------
    def resetUIDs(self, studyUID):
        """Reset SOPInstanceUID, SeriesInstanceUID and StudyInstaceUID (must pass StudyInstaceUID)

        Args:
            studyUID (str): UID - can use str(generate_uid())
        """
        seriesUID = str(generate_uid())
        for k1 in range(len(self)):
            self[k1].SOPInstanceUID = str(generate_uid())
            self[k1].SeriesInstanceUID = seriesUID
            self[k1].StudyInstanceUID = studyUID


    def anonymise(self, anonName, anonPatientID, anon_birthdate=True, remove_private_tags=True):
        """Anonymise series inplace

        Args:
            anonName (str): New anonymise name (can be empty str)
            anonPatientID (str): New anon patient ID, if None then no action taken.
            anon_birthdate (bool, optional): To remove birthdate from tags. Defaults to True.
            remove_private_tags (bool, optional): To remove private tags from dataset. Defaults to True. 
                In some cases this will need to be set False to keep tags necessary for post processing (e.g. some DTI tags). 
                It is up to the user to confirm complete anonymisation in these cases. 
        """
        def PN_callback(ds, data_element):
            """Called from the dataset "walk" recursive function for all data elements."""
            if data_element.VR == "PN":
                data_element.value = 'anonymous'
            if "Institution" in data_element.name:
                data_element.value = 'anonymous'
            if data_element.name == "Patient's Name":
                data_element.value = anonName
            if (data_element.name == "Patient ID") and (anonPatientID is not None):
                data_element.value = anonPatientID

        # Remove patient name and any other person names using a callback to walk over data elements for each ds
        for dataset in self:
            try:
                dataset.walk(PN_callback)
            except TypeError: 
                pass
            # Other Tags
            for name in ['OtherPatientIDs', 'OtherPatientIDsSequence', 'PatientAddress']:
                if name in dataset:
                    delattr(dataset, name)
            if anon_birthdate:
                for name in ['PatientBirthDate']:
                    if name in dataset:
                        dataset.data_element(name).value = ''
            if remove_private_tags:
                try:
                    dataset.remove_private_tags()
                except TypeError:
                    pass


    def writeToOrganisedFileStructure(self, studyOutputDir, seriesOutDirName=None):
        """ Write the series to an organised file structure.
            A hierarchical folder structure rooted at 'studyOutputDir'

        Args:
            studyOutputDir (str): The output directory.
            seriesOutDirName (str, optional): The name of the series output directory. Defaults to None.

        Returns:
            str: The series output directory name.
        """
        self.checkIfShouldUse_SAFE_NAMING()
        ADD_TRANSFERSYNTAX = False
        if seriesOutDirName is None:
            seriesOutDirName = self.getSeriesOutDirName()
        seriesOutputDir = os.path.join(studyOutputDir, seriesOutDirName)
        seriesOutputDirTemp = seriesOutputDir+".WORKING"
        if os.path.isdir(seriesOutputDir):
            os.rename(seriesOutputDir, seriesOutputDirTemp)
        if self.FORCE_READ:
            ADD_TRANSFERSYNTAX = True # need to add if required force read. 
        TO_DECOMPRESS = self.isCompressed()
        for ds in tqdm(self.yieldDataset(), total=len(self), disable=self.HIDE_PROGRESSBAR):
            if TO_DECOMPRESS:
                try:
                    ds.decompress()
                except AttributeError:
                    print(f'Error with file in {seriesOutputDirTemp}')
                    continue
            if ADD_TRANSFERSYNTAX:
                ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
            destFile = dcmTools.writeOut_ds(ds, seriesOutputDirTemp, SAFE_NAMING=self.SAFE_NAME_MODE)
        # ON COMPLETION RENAME OUTPUTDIR
        os.rename(seriesOutputDirTemp, seriesOutputDir)
        return seriesOutputDir


    def _generateFileName(self, tagsToUse, extn):
        if type(tagsToUse) == str:
            fileName = tagsToUse
        else:
            fileName = '_'.join([str(self.getTag(i)) for i in tagsToUse])
        fileName = dcmTools.cleanString(fileName)
        if (len(extn) > 0) and (not extn.startswith('.')):
            extn = '.'+extn
        return fileName+extn


    def writeToNII(self, outputPath, outputNamingTags=('PatientName', 'SeriesNumber', 'SeriesDescription')):
        """Will write DicomSeries as nii.gz file (uses dcm2niix found on local system)

        Args:
            outputPath (str): Directory where to save output (if full pathname, with extn, is given then this is used)
            outputNamingTags (tuple, optional): Tags to use to name output. If outputPath contains extension then not used. Defaults to ('PatientName', 'SeriesNumber', 'SeriesDescription').

        Returns:
            str: Name of files saved
        """
        if outputPath.endswith('nii.gz'):
            outputPath, fileName = os.path.split(outputPath)
        else:
            fileName = self._generateFileName(outputNamingTags, '.nii.gz')
        return dcmTools.writeDirectoryToNII(self.getRootDir(), outputPath, fileName=fileName)


    def writeToVTI(self, outputPath, outputNamingTags=('PatientName', 'SeriesNumber', 'SeriesDescription'), TRUE_ORIENTATION=False):
        """Write DicomSeries as VTK ImageData (`*.vti`)

        Args:
            outputPath (str): Directory to save in or full filename to save
            outputNamingTags (tuple, optional): Tags to use for naming, only used if full outputpath not given. Defaults to ('PatientName', 'SeriesNumber', 'SeriesDescription').
            TRUE_ORIENTATION (bool, optional): To apply matrix to VTI data. Defaults to False.

        Returns:
            str: Name of file saved
        """
        if outputPath.endswith('vti') or outputPath.endswith('pvd'):
            outputPath, fileName = os.path.split(outputPath)
            fileName, _ = os.path.splitext(fileName)
        else:
            fileName = self._generateFileName(outputNamingTags, '')
        vtiDict = self.buildVTIDict(TRUE_ORIENTATION=TRUE_ORIENTATION, outputPath=outputPath)
        return dcmVTKTK.writeVTIDict(vtiDict, outputPath, fileName)


    def writeToVTS(self, outputPath, outputNamingTags=('PatientName', 'SeriesNumber', 'SeriesDescription')):
        """Write DicomSeries as VTK StructuredImageData (`*.vts`)

        Args:
            outputPath (str): Directory to save in or full filename to save.
            outputNamingTags (tuple, optional): Tags to use for naming, only used if full outputpath not given. Defaults to ('PatientName', 'SeriesNumber', 'SeriesDescription').

        Returns:
            str: Name of file saved
        """
        if outputPath.endswith('vts') or outputPath.endswith('pvd'):
            outputPath, fileName = os.path.split(outputPath)
            fileName, _ = os.path.splitext(fileName)
        else:
            fileName = self._generateFileName(outputNamingTags, '')
        vtsDict = self.buildVTSDict(outputPath)
        return dcmVTKTK.fIO.writeVTK_PVD_Dict(vtsDict, outputPath, filePrefix=fileName, fileExtn='vts', BUILD_SUBDIR=True)


    def buildVTSDict(self, outputPath=None):
        """Build a VTK StructuredImageData dictionary from the pixel data.

        Args:
            outputPath (str, optional): The output path. Defaults to None.

        Returns:
            dict: The VTK StructuredImageData dictionary.
        """
        A, patientMeta = self.getPixelDataAsNumpy()
        return dcmVTKTK.arrToVTS(A, patientMeta, self[0], outputPath)


    def buildVTIDict(self, TRUE_ORIENTATION=False, outputPath=None):
        """Build a VTK ImageData dictionary from the pixel data.

        Args:
            TRUE_ORIENTATION (bool, optional): Whether to apply the true orientation to the VTI data. Defaults to False.
            outputPath (str, optional): The output path. Defaults to None.

        Returns:
            dict: The VTK ImageData dictionary. Keys: TriggerTime, Values: VTK ImageData.
        """
        A, patientMeta = self.getPixelDataAsNumpy()
        return dcmVTKTK.arrToVTI(A, patientMeta, self[0], TRUE_ORIENTATION=TRUE_ORIENTATION, outputPath=outputPath)


    @property
    def sliceLocations(self):
        return sorted([float(self.getTag('SliceLocation', i, ifNotFound=0.0)) for i in range(len(self))])


    def getNumberOfSlicesPerVolume(self):
        """Get the number of slices per volume.
        For traditional 2D slice-based DICOM files, this returns the number of unique slice locations.
        For 3D DICOM files, this returns the depth dimension from the pixel data.
        
        Returns:
            int: Number of slices per volume
        """
        if self.has3DPixelData():
            # For 3D DICOM files, return the depth dimension from pixel data
            return self[0].pixel_array.shape[2] if len(self[0].pixel_array.shape) >= 3 else 1
        else:
            # For traditional 2D slice-based DICOM files
            sliceLoc = self.sliceLocations
            return len(set(sliceLoc))


    def getNumberOfTimeSteps(self):
        """Get the number of time steps.
        For traditional 2D slice-based DICOM files, this returns the number of time points per slice location.
        For 3D DICOM files, this returns the number of files (each file is a time point).
        
        Returns:
            int: Number of time steps
        """
        if self.has3DPixelData():
            # For 3D DICOM files, each file represents a time point
            return len(self)
        else:
            # For traditional 2D slice-based DICOM files
            sliceLoc = self.sliceLocations
            sliceLocS = set(sliceLoc)
            return sliceLoc.count(sliceLocS.pop())


    def getSliceNormalVector(self):
        """Get the normal vector of the slice. 
            Calculated from the ImageOrientationPatient tag.

        Returns:
            numpy.ndarray: The normal vector of the slice.
        """
        self.sortBySlice_InstanceNumber()
        try:
            sliceVec = self.getImagePositionPatient_np(-1) - self.getImagePositionPatient_np(0)
        except:
            sliceVec = [0.0,0.0,0.0] # If no slice locations then assume no normal vector (one image)
        if np.linalg.norm(sliceVec) < 1e-9:
            iop = self.getTag('ImageOrientationPatient', ifNotFound=[1.0,0.0,0.0,0.0,1.0,0.0])
            sliceVec = np.cross(iop[:3], iop[3:6])
        else:
            sliceVec = sliceVec / np.linalg.norm(sliceVec)
        return sliceVec


    def doesSliceLocationNorm_Match_IOPNormal(self):
        iop = self.getTag('ImageOrientationPatient')
        iopN = np.cross(iop[:3], iop[3:6])
        sliceLocN = self.getSliceNormalVector()
        tf = [i*j>=0.0 for i,j in zip(iopN, sliceLocN)]
        return all(tf)


    def is3D(self):
        """Check if the DICOM series represents 3D data.
        This can be either:
        1. Multiple 2D slices that form a 3D volume, or
        2. Individual DICOM files containing 3D pixel data
        
        Returns:
            bool: True if the series represents 3D data
        """
        return self.getNumberOfSlicesPerVolume() > 1 or self.has3DPixelData()


    def is4D(self):
        return self.getNumberOfTimeSteps() > 1


    def has3DPixelData(self):
        """Check if the DICOM files contain 3D pixel data (volume data) rather than 2D slices.
        
        Returns:
            bool: True if pixel data has more than 2 dimensions, False if 2D slices
        """
        if len(self) == 0:
            return False
        try:
            firstPixelArray = self[0].pixel_array
            return len(firstPixelArray.shape) > 2
        except AttributeError:
            return False


    def getPixelDataAsNumpy(self):
        """Get pixel data as numpy array organised by slice and time(if present).
            Also return a PatientMatrix object containing meta
            NOTE: Some data is repeated to serve VTK and DICOM conventions
            Handles both 2D slice-based DICOM files and 3D DICOM files

        Returns:
            tuple: 
                numpy array - shape [nRow, nCol, nSlice, nTime], 
                dictionary - keys: Spacing, Origin, ImageOrientationPatient, PatientPosition, Dimensions, Times
        """
        self._loadToMemory()
        
        # Check if we have 3D DICOM files (pixel data with more than 2 dimensions)
        firstPixelArray = self[0].pixel_array
        is3DPixelData = len(firstPixelArray.shape) > 2
        
        if is3DPixelData:
            # Handle 3D DICOM files - each file contains a 3D volume
            print("Detected 3D DICOM files - processing as volume data")
            I, J, K = firstPixelArray.shape[:3]  # First 3 dimensions are spatial
            N = len(self)  # Number of time steps = number of files
            
            # Check if we have a 4th dimension (e.g., RGB channels)
            if len(firstPixelArray.shape) == 4:
                # Handle RGB/RGBA data
                channels = firstPixelArray.shape[3]
                A = np.zeros((I, J, K, N, channels), dtype=firstPixelArray.dtype)
                for n in range(N):
                    A[:, :, :, n, :] = self[n].pixel_array
                # Reshape to standard format: (I, J, K*N, channels) for compatibility
                A = A.reshape((I, J, K*N, channels))
            else:
                # Standard 3D data
                A = np.zeros((I, J, K, N), dtype=firstPixelArray.dtype)
                for n in range(N):
                    A[:, :, :, n] = self[n].pixel_array
                # Reshape to standard format: (I, J, K*N) for compatibility
                A = A.reshape((I, J, K*N))
                
        else:
            # Handle traditional 2D slice-based DICOM files
            I, J = int(self.getTag('Rows')), int(self.getTag('Columns'))
            K = int(self.getNumberOfSlicesPerVolume())
            self.sortBySlice_InstanceNumber()  # This takes care of order - slices grouped, then time for each slice 
            N = self.getNumberOfTimeSteps()
            
            thisDType = firstPixelArray.dtype
            thisShape = firstPixelArray.shape
            A = np.zeros((I, J, K, N), dtype=thisDType)
            if (K*N) != len(self):
                raise FileNotFoundError(f"Missing some DICOM files.")
            
            c0 = 0
            for k1 in range(K):
                for k2 in range(N):
                    iA = self[c0].pixel_array
                    A[:, :, k1, k2] = iA
                    c0 += 1
        
        patientMeta = dcmVTKTK.PatientMeta()
        patientMeta.initFromDicomSeries(self)
        return A, patientMeta


    def getScanDuration_secs(self):
        try:
            return self.getTag(0x0019105a, ifNotFound=0.0) / 1000000.0
        except AttributeError:
            return 0.0


    def _getPixelSpacing(self):
        """Get pixel spacing, handling both 2D and 3D DICOM files.
        For 3D files, checks SharedFunctionalGroupsSequence if standard tags not available.
        """
        if self.has3DPixelData():
            # For 3D DICOM files, try standard tags first, then SharedFunctionalGroupsSequence
            pixelSpacing = self.getTag('PixelSpacing', ifNotFound=None)
            if pixelSpacing is not None:
                return [float(i) for i in pixelSpacing]
            
            # Try to get from SharedFunctionalGroupsSequence
            pixelSpacing = self._getPixelSpacingFromSharedFunctionalGroups()
            if pixelSpacing is not None:
                return pixelSpacing
        
        # Fallback to standard method or default
        return [float(i) for i in self.getTag('PixelSpacing', ifNotFound=[0.0,0.0])]


    def getDeltaRow(self):
        """Get the pixel spacing in the row direction.

        Returns:
            float: The pixel spacing in the row direction in mm.
        """
        return self._getPixelSpacing()[0]


    def getDeltaCol(self):
        """Get the pixel spacing in the column direction.

        Returns:
            float: The pixel spacing in the column direction in mm.
        """
        return self._getPixelSpacing()[1]


    def getDeltaSlice(self):
        """Get the slice spacing of the series.
            For traditional 2D slice-based DICOM files:
                If there is only one slice (or single slice CINE) then the slice spacing is taken from the SpacingBetweenSlices or SliceThickness (if not present) tag.
                If there is more than one slice then the slice spacing is taken from the mean of the slice locations.
            For 3D DICOM files:
                The slice spacing is prioritized as: SpacingBetweenSlices > SliceThickness > SharedFunctionalGroupsSequence > default.
                This ensures proper 3D volume construction by using the actual distance between slices rather than overlapping slice thickness.

        Returns:
            float: The slice spacing in mm needed to construct a proper 3D volume.
        """
        if self.has3DPixelData():
            # For 3D DICOM files, try to get slice spacing from various locations
            dZ = None
            
            # First try SpacingBetweenSlices (preferred for 3D volume construction)
            dZ = self.getTag('SpacingBetweenSlices', ifNotFound=None)
            
            # If not found, try SliceThickness as fallback
            if dZ is None:
                dZ = self.getTag('SliceThickness', ifNotFound=None)
            
            # If still not found, try to extract from SharedFunctionalGroupsSequence
            if dZ is None:
                dZ = self._getSliceSpacingFromSharedFunctionalGroups()
            
            if dZ is not None:
                return float(dZ)
            else:
                # If no explicit slice spacing is available, use a reasonable default
                print("Warning: No slice spacing information found for 3D DICOM file, using default")
                return 1.0  # Default 1mm spacing
        else:
            # For traditional 2D slice-based DICOM files
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


    def _getSliceSpacingFromSharedFunctionalGroups(self):
        """Extract slice spacing from SharedFunctionalGroupsSequence for 3D DICOM files.
        Also checks PerFrameFunctionalGroupsSequence if SharedFunctionalGroupsSequence doesn't have the data.
        This handles the nested structure where slice spacing information is stored.
        Prioritizes SpacingBetweenSlices over SliceThickness for proper 3D volume construction.
        
        Returns:
            float or None: The slice spacing in mm if found, None otherwise
        """
        try:
            # First check SharedFunctionalGroupsSequence
            if hasattr(self[0], 'SharedFunctionalGroupsSequence') and self[0].SharedFunctionalGroupsSequence:
                shared_seq = self[0].SharedFunctionalGroupsSequence[0]
                
                # Check for PixelMeasuresSequence within SharedFunctionalGroupsSequence
                if hasattr(shared_seq, 'PixelMeasuresSequence') and shared_seq.PixelMeasuresSequence:
                    pixel_measures = shared_seq.PixelMeasuresSequence[0]
                    
                    # Try to get SpacingBetweenSlices first (preferred for 3D volume construction)
                    if hasattr(pixel_measures, 'SpacingBetweenSlices'):
                        return float(pixel_measures.SpacingBetweenSlices)
                    
                    # Try SliceThickness as fallback
                    if hasattr(pixel_measures, 'SliceThickness'):
                        return float(pixel_measures.SliceThickness)
                
                # Check for PlanePositionSequence as another possible location
                if hasattr(shared_seq, 'PlanePositionSequence') and shared_seq.PlanePositionSequence:
                    plane_pos = shared_seq.PlanePositionSequence[0]
                    if hasattr(plane_pos, 'SliceThickness'):
                        return float(plane_pos.SliceThickness)
            
            # If not found in SharedFunctionalGroupsSequence, check PerFrameFunctionalGroupsSequence
            if hasattr(self[0], 'PerFrameFunctionalGroupsSequence') and self[0].PerFrameFunctionalGroupsSequence:
                # Check the first frame for slice spacing
                per_frame_seq = self[0].PerFrameFunctionalGroupsSequence[0]
                
                # Check for PixelMeasuresSequence within PerFrameFunctionalGroupsSequence
                if hasattr(per_frame_seq, 'PixelMeasuresSequence') and per_frame_seq.PixelMeasuresSequence:
                    pixel_measures = per_frame_seq.PixelMeasuresSequence[0]
                    
                    # Try to get SpacingBetweenSlices first (preferred for 3D volume construction)
                    if hasattr(pixel_measures, 'SpacingBetweenSlices'):
                        return float(pixel_measures.SpacingBetweenSlices)
                    
                    # Try SliceThickness as fallback
                    if hasattr(pixel_measures, 'SliceThickness'):
                        return float(pixel_measures.SliceThickness)
                
                # Check for PlanePositionSequence as another possible location
                if hasattr(per_frame_seq, 'PlanePositionSequence') and per_frame_seq.PlanePositionSequence:
                    plane_pos = per_frame_seq.PlanePositionSequence[0]
                    if hasattr(plane_pos, 'SliceThickness'):
                        return float(plane_pos.SliceThickness)
            
            return None
        except (AttributeError, IndexError, ValueError, TypeError):
            # If any error occurs during extraction, return None
            return None


    def _getPixelSpacingFromSharedFunctionalGroups(self):
        """Extract pixel spacing from SharedFunctionalGroupsSequence for 3D DICOM files.
        Also checks PerFrameFunctionalGroupsSequence if SharedFunctionalGroupsSequence doesn't have the data.
        
        Returns:
            list or None: [row_spacing, col_spacing] in mm if found, None otherwise
        """
        try:
            # First check SharedFunctionalGroupsSequence
            if hasattr(self[0], 'SharedFunctionalGroupsSequence') and self[0].SharedFunctionalGroupsSequence:
                shared_seq = self[0].SharedFunctionalGroupsSequence[0]
                
                if hasattr(shared_seq, 'PixelMeasuresSequence') and shared_seq.PixelMeasuresSequence:
                    pixel_measures = shared_seq.PixelMeasuresSequence[0]
                    
                    if hasattr(pixel_measures, 'PixelSpacing'):
                        return [float(i) for i in pixel_measures.PixelSpacing]
            
            # If not found in SharedFunctionalGroupsSequence, check PerFrameFunctionalGroupsSequence
            if hasattr(self[0], 'PerFrameFunctionalGroupsSequence') and self[0].PerFrameFunctionalGroupsSequence:
                # Check the first frame for pixel spacing
                per_frame_seq = self[0].PerFrameFunctionalGroupsSequence[0]
                
                if hasattr(per_frame_seq, 'PixelMeasuresSequence') and per_frame_seq.PixelMeasuresSequence:
                    pixel_measures = per_frame_seq.PixelMeasuresSequence[0]
                    
                    if hasattr(pixel_measures, 'PixelSpacing'):
                        return [float(i) for i in pixel_measures.PixelSpacing]
            
            return None
        except (AttributeError, IndexError, ValueError, TypeError):
            return None


    def _getImagePositionPatientFromSharedFunctionalGroups(self):
        """Extract image position patient from SharedFunctionalGroupsSequence for 3D DICOM files.
        Also checks PerFrameFunctionalGroupsSequence if SharedFunctionalGroupsSequence doesn't have the data.
        
        Returns:
            list or None: [x, y, z] position in mm if found, None otherwise
        """
        try:
            # First check SharedFunctionalGroupsSequence
            if hasattr(self[0], 'SharedFunctionalGroupsSequence') and self[0].SharedFunctionalGroupsSequence:
                shared_seq = self[0].SharedFunctionalGroupsSequence[0]
                
                # Check for PlanePositionSequence
                if hasattr(shared_seq, 'PlanePositionSequence') and shared_seq.PlanePositionSequence:
                    plane_pos = shared_seq.PlanePositionSequence[0]
                    if hasattr(plane_pos, 'ImagePositionPatient'):
                        return [float(i) for i in plane_pos.ImagePositionPatient]
            
            # If not found in SharedFunctionalGroupsSequence, check PerFrameFunctionalGroupsSequence
            if hasattr(self[0], 'PerFrameFunctionalGroupsSequence') and self[0].PerFrameFunctionalGroupsSequence:
                # Check the first frame for image position
                per_frame_seq = self[0].PerFrameFunctionalGroupsSequence[0]
                
                # Check for PlanePositionSequence
                if hasattr(per_frame_seq, 'PlanePositionSequence') and per_frame_seq.PlanePositionSequence:
                    plane_pos = per_frame_seq.PlanePositionSequence[0]
                    if hasattr(plane_pos, 'ImagePositionPatient'):
                        return [float(i) for i in plane_pos.ImagePositionPatient]
            
            return None
        except (AttributeError, IndexError, ValueError, TypeError):
            return None


    def _getImageOrientationPatientFromSharedFunctionalGroups(self):
        """Extract image orientation patient from SharedFunctionalGroupsSequence for 3D DICOM files.
        Also checks PerFrameFunctionalGroupsSequence if SharedFunctionalGroupsSequence doesn't have the data.
        
        Returns:
            list or None: [x1, y1, z1, x2, y2, z2] orientation if found, None otherwise
        """
        try:
            # First check SharedFunctionalGroupsSequence
            if hasattr(self[0], 'SharedFunctionalGroupsSequence') and self[0].SharedFunctionalGroupsSequence:
                shared_seq = self[0].SharedFunctionalGroupsSequence[0]
                
                # Check for PlaneOrientationSequence
                if hasattr(shared_seq, 'PlaneOrientationSequence') and shared_seq.PlaneOrientationSequence:
                    plane_orient = shared_seq.PlaneOrientationSequence[0]
                    if hasattr(plane_orient, 'ImageOrientationPatient'):
                        return [float(i) for i in plane_orient.ImageOrientationPatient]
            
            # If not found in SharedFunctionalGroupsSequence, check PerFrameFunctionalGroupsSequence
            if hasattr(self[0], 'PerFrameFunctionalGroupsSequence') and self[0].PerFrameFunctionalGroupsSequence:
                # Check the first frame for image orientation
                per_frame_seq = self[0].PerFrameFunctionalGroupsSequence[0]
                
                # Check for PlaneOrientationSequence
                if hasattr(per_frame_seq, 'PlaneOrientationSequence') and per_frame_seq.PlaneOrientationSequence:
                    plane_orient = per_frame_seq.PlaneOrientationSequence[0]
                    if hasattr(plane_orient, 'ImageOrientationPatient'):
                        return [float(i) for i in plane_orient.ImageOrientationPatient]
            
            return None
        except (AttributeError, IndexError, ValueError, TypeError):
            return None


    def hasVariableSliceThickness(self):
        """Check if the 3D DICOM file has variable slice thickness.
        This is important because variable slice thickness is not supported in the current VTK conversion.
        
        Returns:
            bool: True if variable slice thickness is detected, False otherwise
        """
        if not self.has3DPixelData():
            return False  # Only relevant for 3D DICOM files
        
        try:
            # Check if there are multiple frames with different slice thickness values
            if hasattr(self[0], 'SharedFunctionalGroupsSequence') and self[0].SharedFunctionalGroupsSequence:
                shared_seq = self[0].SharedFunctionalGroupsSequence[0]
                
                # Check for PerFrameFunctionalGroupsSequence which indicates variable parameters
                if hasattr(shared_seq, 'PerFrameFunctionalGroupsSequence'):
                    per_frame_seq = shared_seq.PerFrameFunctionalGroupsSequence
                    
                    # If there are multiple frames, check if they have different slice thickness
                    if len(per_frame_seq) > 1:
                        thicknesses = set()
                        for frame in per_frame_seq:
                            if hasattr(frame, 'PixelMeasuresSequence') and frame.PixelMeasuresSequence:
                                pixel_measures = frame.PixelMeasuresSequence[0]
                                if hasattr(pixel_measures, 'SliceThickness'):
                                    thicknesses.add(float(pixel_measures.SliceThickness))
                        
                        # If we found multiple different thickness values, it's variable
                        if len(thicknesses) > 1:
                            print(f"Warning: Variable slice thickness detected: {thicknesses}")
                            return True
            
            return False
        except (AttributeError, IndexError, ValueError, TypeError):
            return False


    def get3DSpacing(self):
        """Get the complete 3D spacing information (row, column, slice) in mm.
        This method is particularly useful for 3D DICOM files and VTK volume conversion.
        
        Returns:
            tuple: (deltaRow, deltaCol, deltaSlice) in mm
        """
        if self.has3DPixelData():
            # For 3D DICOM files, get spacing from enhanced methods
            deltaRow = self.getDeltaRow()
            deltaCol = self.getDeltaCol()
            deltaSlice = self.getDeltaSlice()
            
            return (deltaRow, deltaCol, deltaSlice)
        else:
            # For traditional 2D slice-based DICOM files
            return (self.getDeltaRow(), self.getDeltaCol(), self.getDeltaSlice())


    def hasExplicitSpacingInfo(self):
        """Check if the DICOM series has explicit spacing information.
        This is useful for determining the quality of spacing data, especially for 3D volumes.
        
        Returns:
            bool: True if explicit spacing information is available, False otherwise
        """
        if self.has3DPixelData():
            # For 3D DICOM files, check for explicit spacing information
            hasSliceThickness = self.getTag('SliceThickness', ifNotFound=None) is not None
            hasSpacingBetweenSlices = self.getTag('SpacingBetweenSlices', ifNotFound=None) is not None
            hasPixelSpacing = self.getTag('PixelSpacing', ifNotFound=None) is not None
            hasImagePositionPatient = self.getTag('ImagePositionPatient', ifNotFound=None) is not None
            hasImageOrientationPatient = self.getTag('ImageOrientationPatient', ifNotFound=None) is not None
            
            # Also check SharedFunctionalGroupsSequence
            hasSharedFunctionalGroups = (self._getSliceSpacingFromSharedFunctionalGroups() is not None or
                                       self._getPixelSpacingFromSharedFunctionalGroups() is not None or
                                       self._getImagePositionPatientFromSharedFunctionalGroups() is not None or
                                       self._getImageOrientationPatientFromSharedFunctionalGroups() is not None)
            
            return (hasSliceThickness or hasSpacingBetweenSlices or hasPixelSpacing or 
                   hasImagePositionPatient or hasImageOrientationPatient or hasSharedFunctionalGroups)
        else:
            # For traditional 2D slice-based DICOM files, check for slice location information
            return len(self.sliceLocations) > 1


    def getTemporalResolution(self):
        """Get the temporal resolution of the series.
            NominalInterval / CardiacNumberOfImages
            If CardiacNumberOfImages is not present then 1 is assumed.
            If NominalInterval is not present then 0.0 is assumed.

        Returns:
            float: The temporal resolution in seconds.
        """
        try:
            return float(self.getTag('NominalInterval', ifNotFound=0.0)/self.getTag('CardiacNumberOfImages', ifNotFound=1))
        except ZeroDivisionError:
            return 0       


    def getTemporalResolution_TR_VPS(self):
        return float(self.getTag('RepetitionTime', ifNotFound=0.0)*self.getTag(0x00431007, ifNotFound=1.0))


    def getManufacturer(self):
        return self.getTag(0x00080070, ifNotFound='Unknown')


    def IS_GE(self):
        return self.getManufacturer().lower().startswith('ge')


    def IS_SIEMENS(self):
        return self.getManufacturer().lower().startswith('siemens')


    def IS_PHILIPS(self):
        return self.getManufacturer().lower().startswith('philips')


    def IS_BRUKER(self):
        return self.getManufacturer().lower().startswith('bruker')


    def getPulseSequenceName(self):
        if self.IS_GE():
            return self.getTag(0x0019109c)
        else:
            return self.getTag(0x00180024)


    def getVENC(self):
        """Get VENC value in mm/s
        """
        if self.IS_GE():
            return float(self.getTag(0x001910cc))
        elif self.IS_SIEMENS():
            vencStr = self.getTag(0x00511014)
            return float(vencStr.split("_")[0].replace("v", "")) * 10.0
        else: # TODO - add other vendors
            return None


    def getInternalPulseSequenceName(self):
        return self.getTag(0x0019109e)


    def getSeriesDescription(self):
        return self.getTag('SeriesDescription')


    def getSeriesInfoDict(self, extraTags=[], EXTRA_TAGS=[]):
        """Get a dictionary of detailed series information including internally calculated values:
            - ScanDuration
            - nTime
            - nRow
            - nCols
            - dRow - pixel spacing in row direction
            - dCol - pixel spacing in column direction
            - dSlice - slice thickness
            - dTime - temporal resolution
            - SpacingBetweenSlices
            - FlipAngle
            - HeartRate
            - EchoTime
            - RepetitionTime
            - MagneticFieldStrength

        Args:
            EXTRA_TAGS (list, optional): DEPRECIATED - use extraTags instead.
            extraTags (list, optional): Additional tags to add to the dictionary. Defaults to [].

        Returns:
            dict: A dictionary of series information.
        """
        outDict = {'ScanDuration':self.getScanDuration_secs(),
            'nTime':self.getTag('CardiacNumberOfImages'),
            'nRow':self.getTag('Rows'),
            'nCols':self.getTag('Columns'),
            'dRow':self.getDeltaRow(),
            'dCol':self.getDeltaCol(),
            'dSlice':self.getTag('SliceThickness'),
            'dTime': self.getTemporalResolution(),
            'SpacingBetweenSlices':self.getTag('SpacingBetweenSlices'),
            'FlipAngle':self.getTag('FlipAngle'),
            'HeartRate':self.getTag('HeartRate'),
            'EchoTime':self.getTag('EchoTime'),
            'RepetitionTime':self.getTag('RepetitionTime'),
            'PulseSequenceName':self.getPulseSequenceName(),
            'MagneticFieldStrength': self.getTag('MagneticFieldStrength'),
            'InternalPulseSequenceName':self.getInternalPulseSequenceName(),
            'ReconstructionDiameter': self.getTag(0x00181100),
            'AcquisitionMatrix': str(self.getTag(0x00181310)),
            'Manufacturer': self.getTag("Manufacturer"),
            'ManufacturerModelName': self.getTag("ManufacturerModelName"),
            'SoftwareVersions': str(self.getTag(0x00181020)),}
        if len(EXTRA_TAGS) > 0:
            print(f"WARNING: EXTRA_TAGS is DEPRECIATED and will be removed in future versions. Use 'extraTags' instead.")
        extraTags += EXTRA_TAGS
        for extraTag in extraTags:
            if extraTag not in outDict.keys():
                try:
                    outDict[extraTag] = self.getTag(extraTag)
                except:
                    outDict[extraTag] = 'Unknown'
        outDict['ImagesInAcquisition'] = len(self)
        try:
            outDict['AcquiredResolution'] = float(outDict['ReconstructionDiameter']) / float(max(self.getTag(0x00181310)))
        except (TypeError, ValueError):
            outDict['AcquiredResolution'] = f"{outDict['dRow']},{outDict['dCol']}"
        try:
            outDict['AcquiredTemporalResolution'] = self.getTemporalResolution_TR_VPS()
        except (TypeError, ValueError):
            outDict['AcquiredTemporalResolution'] = 0.0
        for i in outDict.keys():
            try:
                if ',' in outDict[i]:
                    outDict[i] = str(outDict[i])
            except TypeError:
                pass
        return outDict


    def checkIfShouldUse_SAFE_NAMING(self, se_instance_set=None):
        """Checks is should name dicoms based on UIDs or if a SE-number, Instace-number naming convention is possible.
        Basically checks if there would be any overlaps. 

        Args:
            se_instance_set (set, optional): A set of instance file names (without extension). Defaults to None.
        """
        if self.SAFE_NAME_MODE:
            return
        if se_instance_set is None:
            se_instance_set = set()
        for k1 in range(len(self)):
            se = self.getTag('SeriesNumber', dsID=k1, ifNotFound='unknown')
            instance = self.getTag('InstanceNumber', dsID=k1, ifNotFound='unknown')
            se_instance_str = f"{se}_{instance}"
            if se_instance_str in se_instance_set:
                self.SAFE_NAME_MODE = True
                break
            se_instance_set.add(se_instance_str)


    def getStudyOutputDir(self, rootDir='', studyPrefix=''):
        """Generate a study output directory name based on config setting STUDY_NAMINF_TAG_LIST

        Args:
            rootDir (str, optional): The root directory to build study directory in. Defaults to ''.
            studyPrefix (str, optional): Optional prefix to add to naming. Defaults to ''.

        Returns:
            str: Path to directory at Study level where dicoms will be written
        """
        suffix = ''
        if self.SAFE_NAME_MODE:
            suffix += f"{self.getTag('StudyInstanceUID')}"
        for iTag in SpydcmTK_config.STUDY_NAMING_TAG_LIST:
            iVal = self.getTag(iTag, ifNotFound='', convertToType=str)
            if len(iVal) > 0:
                suffix += '_'+iVal
        return os.path.join(rootDir, dcmTools.cleanString(studyPrefix+suffix))


    def overviewImage(self, outputFileName=None, RETURN_FIG=False):
        """
        Build an overview image of the series.

        Args:
            outputFileName (str, optional): The name of the file to save the overview image to. Defaults to None.
            RETURN_FIG (bool, optional): Whether to return the figure. Defaults to False.

        Returns:
            if RETURN_FIG is True, returns the figure.
            if outputFileName is not None, saves the figure to the file and returns the file name.
            otherwise, displays the figure.
        """
        A = self.getPixelDataAsNumpy()[0]
        i,j,k,n = A.shape
        if k > 1:
            iSlice = k//2
            Image_list = [A[:,:,iSlice,0], np.rot90(A[i//2,:,:,0],3), np.rot90(A[:,j//2,:,0],3)]
        else:
            Image_list = [A[:,:,0,0]]
        fig, axs = plt.subplots(1, len(Image_list))
        if len(Image_list) == 1:
            axs = [axs]  # Convert single Axes to list for consistent indexing
        for i, iA in enumerate(Image_list):
            axs[i].imshow(iA, cmap='gray')
            axs[i].axis('off')
        fig.tight_layout()
        if RETURN_FIG:
            return fig
        if outputFileName is not None:
            plt.savefig(outputFileName)
            plt.close()
            return outputFileName
        else:
            plt.show()
        


# ================================================================================================================
# DicomStudy
# ================================================================================================================
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
        return cls(dStudyList[0], OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR)

    @classmethod
    def setFromDirectory(cls, dirName, OVERVIEW=False, HIDE_PROGRESSBAR=False, FORCE_READ=False, ONE_FILE_PER_DIR=False):
        dicomDict = dcmTools.organiseDicomHeirarchyByUIDs(dirName, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ, ONE_FILE_PER_DIR=ONE_FILE_PER_DIR, OVERVIEW=OVERVIEW)
        obj = DicomStudy.setFromDictionary(dicomDict, OVERVIEW=OVERVIEW, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ)
        for iSeries in obj:
            iSeries.NOT_FULLY_LOADED = ONE_FILE_PER_DIR
        return obj


    def __str__(self):
        return '%s: %s: %d series, %d dicoms'%(self.getTag('PatientName'),
                                                self.getTag('SeriesDescription'),
                                                len(self),
                                                self.getNumberOfDicoms())


    def setSafeNameMode(self):
        for iSeries in self:
            iSeries.SAFE_NAME_MODE = True

    def resetUIDs(self, studyUID=None):
        if studyUID is None:
            studyUID = str(generate_uid())
        for i in self:
            i.resetUIDs(studyUID)

    def filterByTag(self, tagName, tagValue):
        return DicomStudy([i for i in self if i.getTag(tagName) == tagValue], 
                           OVERVIEW=self.OVERVIEW, 
                           HIDE_PROGRESSBAR=self.HIDE_PROGRESSBAR)

    def anonymise(self, anonName, anonPatientID, anonBirthdate=True, removePrivateTags=True):
        for i in self:
            i.anonymise(anonName, anonPatientID, anonBirthdate, removePrivateTags)

    def getTag(self, tag, seriesID=0, instanceID=0, ifNotFound='Unknown', convertToType=None):
        return self[seriesID].getTag(tag, dsID=instanceID, ifNotFound=ifNotFound, convertToType=convertToType)

    def getTagListAndNames(self, tagList, seriesID=0, instanceID=0):
        return self[seriesID].getTagListAndNames(tagList, dsID=instanceID)

    def setTags_all(self, tag, value):
        for i in self:
            i.setTags_all(tag, value)

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

    def getStudyOverview(self, tagList=SpydcmTK_config.STUDY_OVERVIEW_TAG_LIST):
        return self.getTagListAndNames(tagList)

    def getPatientOverview(self, tagList=SpydcmTK_config.SUBJECT_OVERVIEW_TAG_LIST):
        return self.getTagListAndNames(tagList)

    def getSeriesByID(self, ID):
        ID = int(ID)
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


    def getSeriesByTag(self, tag, value, convertToType=None):
        for iSeries in self:
            if iSeries.getTag(tag, convertToType=convertToType) == value:
                return iSeries


    def mergeSeriesVolumesWithTime(self):
        """ Convenience method to sort a 4D series by volumes then time.
        """
        nTimes = len(self)
        triggerTimes = sorted([self.getTag('TriggerTime', seriesID=i, instanceID=0, ifNotFound=0, convertToType=float) for i in range(nTimes)])
        nPerTime = len(self.getSeriesByTag('TriggerTime', triggerTimes[0], convertToType=float))
        sorted_ds_list = [None for _ in range(nTimes*nPerTime)]
        # print(f"Merging series: n times: {nTimes}, N per time: {nPerTime}")
        for i in range(nTimes):
            iSeries = self.getSeriesByTag('TriggerTime', triggerTimes[i], convertToType=float)
            iSeries.sortByInstanceNumber()
            for k1, iDS in enumerate(iSeries):
                if nPerTime > 1:
                    thisID = i + (nTimes * k1)
                else:
                    thisID = i
                iDS.InstanceNumber = thisID + 1 # 1 index
                sorted_ds_list[thisID] = iDS
        if sorted_ds_list.count(None) != 0:
            raise ValueError('Missing some volumes')
        return DicomSeries(sorted_ds_list)


    def writeFDQ(self, seriesNumber_list, outputFileName, velArrayName,
                phase_factors=None, phase_offsets=None, working_dir=None, VERBOSE=True):
        """
        Writes a 4D-flow output as VTS format.

        Args:
            seriesNumber_list (list): A list of series numbers identifying mag and 3 phase series. Mag, X, Y, Z phases
            outputFileName (str): The name of the pvd file to write to.

        Returns:
            str: the pvd file written
        """
        # Check phase factors and offsets:
        if phase_factors is not None:
            if len(phase_factors) != 3:
                raise ValueError(f"phase_factors should be length 3")
        else:
            if self.getSeriesBySeriesNumber(seriesNumber_list[1]).IS_GE():
                phase_factors = [1.0, 1.0, -1.0]
                phase_offsets = [0.0, 0.0, 0.0]
            elif self.getSeriesBySeriesNumber(seriesNumber_list[1]).IS_SIEMENS():
                venc =  self.getSeriesBySeriesNumber(seriesNumber_list[1]).getVENC()
                c = -1.0 * venc
                m = (venc * 2.0) / 4096.0 # TODO: Check this
                phase_factors = [m, m, m]
                phase_offsets = [c, c, c]
            elif self.getSeriesBySeriesNumber(seriesNumber_list[1]).IS_PHILIPS(): # TODO: Check this
                phase_factors = [1.0, 1.0, 1.0]
                phase_offsets = [0.0, 0.0, 0.0]
            else:
                raise ValueError(f"Unknown scanner type")
        if phase_offsets is not None:
            if len(phase_offsets) != 3:
                raise ValueError(f"phase_offsets should be length 3")
        else:
            phase_offsets = [0.0, 0.0, 0.0]
        
        if working_dir is None:
            working_dir = os.path.dirname(outputFileName)
        intermediate_pvds = {}
        
        # Create process pool and process all directories in parallel
        origHIDE = self.HIDE_PROGRESSBAR
        self.HIDE_PROGRESSBAR = True
        with Pool() as pool:
            args = [(self.getSeriesBySeriesNumber(iSeriesNum), label, working_dir)
                    for iSeriesNum, label in zip(seriesNumber_list, ["MAG", "PX", "PY", "PZ"])]
            
            results = pool.map(_process_series_for_fdq, args)
        
        # Collect results into intermediate_pvds dictionary
        intermediate_pvds = dict(results)
        
        if VERBOSE: print(f"Combining to final 4D-flow PVD")
        fOut = dcmVTKTK.mergePhaseSeries4D(intermediate_pvds["MAG"], 
                                        [intermediate_pvds["PX"],
                                        intermediate_pvds["PY"],
                                        intermediate_pvds["PZ"]], 
                                        outputFileName, 
                                        phase_factors=phase_factors,
                                        phase_offsets=phase_offsets,
                                        scale_factor=0.001,
                                        velArrayName=velArrayName,
                                        DEL_ORIGINAL=True)
        self.HIDE_PROGRESSBAR = origHIDE
        if VERBOSE: print(f"Written {fOut}")
        return fOut


    def tagsToDictionary(self):
        dict_out = {}
        for i in self:
            dd = i.tagsToDictionary()
            dict_out[i.getTag("SeriesInstanceUID")] = dd
        return dict_out
    

    def tagsToJson(self, jsonFileOut):
        dict_out = self.tagsToDictionary()
        dcmTools.writeDictionaryToJSON(jsonFileOut, dict_out)
        return jsonFileOut


    def getStudySummaryDict(self, extraTags=[]):
        pt,pv = self.getPatientOverview(tagList=SpydcmTK_config.SUBJECT_OVERVIEW_TAG_LIST+extraTags)
        studySummaryDict = dict(zip(pt, pv))
        st,sv = self.getStudyOverview(tagList=SpydcmTK_config.STUDY_OVERVIEW_TAG_LIST+extraTags)
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
            seriesStr_ = '\n\n'+seriesHeader+'\n    '+'\n    '.join(seriesStr)
            strOut = f"SUBJECT: {patStr}"
            strOut += f"\nSTUDY: {studyStr}"
            strOut += seriesStr_
            return strOut
        else:
            strOut = f"SUBJECT: {patStr}"
            strOut += f"\nSTUDY: {studyStr}"
            return strOut


    def getTagValuesList(self, tagList, RETURN_STR=False):
        output = []
        for i in self:
            output += i.getTagValuesList(tagList, False)
        if RETURN_STR:
            return dcmTools._tagValuesListToString(tagList, output)
        return output


    def writeToOrganisedFileStructure(self, patientOutputDir, studyPrefix=''):
        """Write the study to an organised file structure.

        Args:
            patientOutputDir (str): The output root directory (a Study output directory will be built here).
            studyPrefix (str, optional): The prefix of the study output directory. Defaults to ''.

        Returns:
            str: The study output directory name.
        """
        self.checkIfShouldUse_SAFE_NAMING()
        try:
            studyOutputDir = self[0].getStudyOutputDir(patientOutputDir, studyPrefix)
        except IndexError: # Study is empty
            return None
        studyOutputDirTemp = studyOutputDir+".WORKING"
        if os.path.isdir(studyOutputDir):
            os.rename(studyOutputDir, studyOutputDirTemp)
        for iSeries in self:
            iSeries.writeToOrganisedFileStructure(studyOutputDirTemp)
        os.rename(studyOutputDirTemp, studyOutputDir)
        return studyOutputDir


    def writeToZipArchive(self, patientOutputDir, CLEAN_UP=True):
        """Write the study to a zip archive. Study is written to disk then zipped.

        Args:
            patientOutputDir (str): The output directory.
            CLEAN_UP (bool, optional): Whether to clean up the original written directory. Defaults to True.

        Returns:
            str: The zip file written.
        """
        studyOutputDir = self.writeToOrganisedFileStructure(patientOutputDir)
        if studyOutputDir is None:
            return
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
                    raise IOError("ERROR READING DICOMS: SPYDCMTK capable to read dicom files from directory, zip, tar or tar.gz\n")

    @classmethod
    def setFromDirectory(cls, dirName, OVERVIEW=False, HIDE_PROGRESSBAR=False, FORCE_READ=False, ONE_FILE_PER_DIR=False, extn_filter=None):
        dicomDict = dcmTools.organiseDicomHeirarchyByUIDs(dirName, 
                                                         HIDE_PROGRESSBAR=HIDE_PROGRESSBAR, 
                                                         FORCE_READ=FORCE_READ, 
                                                         ONE_FILE_PER_DIR=ONE_FILE_PER_DIR, 
                                                         OVERVIEW=OVERVIEW,
                                                         extn_filter=extn_filter)
        obj = ListOfDicomStudies.setFromDcmDict(dicomDict, OVERVIEW, HIDE_PROGRESSBAR, FORCE_READ=FORCE_READ)
        for iStudy in obj:
            for iSeries in iStudy:
                iSeries.NOT_FULLY_LOADED = ONE_FILE_PER_DIR
        return obj

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

    def resetUIDs(self):
        for i in self:
            i.resetUIDs()

    def anonymise(self, anonName, anonPatientID, anonBirthdate=True, removePrivateTags=True):
        for i in self:
            i.anonymise(anonName, anonPatientID, anonBirthdate, removePrivateTags)

    def isCompressed(self):
        return self[0].isCompressed()

    def getNumberOfDicoms(self):
        return sum([i.getNumberOfDicoms() for i in self])

    def filterByTag(self, tagName, tagValue):
        return ListOfDicomStudies([i for i in self if i.getTag(tagName) == tagValue], 
                                  OVERVIEW=self.OVERVIEW, 
                                  HIDE_PROGRESSBAR=self.HIDE_PROGRESSBAR)

    def writeToOrganisedFileStructure(self, outputRootDir):
        outDirs = []
        for iStudy in self:
            suffix = ''
            for iTag in SpydcmTK_config.SUBJECT_NAMING_TAG_LIST:
                iVal = iStudy.getTag(iTag, ifNotFound='', convertToType=str)
                if len(iVal) > 0:
                    suffix += '_'+iVal
            if len(suffix) > 0:
                ioutputRootDir = os.path.join(outputRootDir, dcmTools.cleanString(suffix))
            else:
                ioutputRootDir = outputRootDir

            ooD = iStudy.writeToOrganisedFileStructure(ioutputRootDir)
            if ooD is not None:
                outDirs.append(ooD)
        return outDirs

    def writeToZipArchive(self, outputRootDir, CLEAN_UP=True):
        outDirs = []
        for iStudy in self:
            ooD = iStudy.writeToZipArchive(outputRootDir, CLEAN_UP=CLEAN_UP)
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

    def printSummaryTable(self):
        overviewList = self.getTableOfTagValues(SpydcmTK_config.SUBJECT_OVERVIEW_TAG_LIST+SpydcmTK_config.STUDY_OVERVIEW_TAG_LIST, False)
        print(','.join(SpydcmTK_config.SUBJECT_OVERVIEW_TAG_LIST+SpydcmTK_config.STUDY_OVERVIEW_TAG_LIST))
        for i in overviewList:
            print(','.join(i))

    def getStudyByDate(self, date_str):
        return self.getStudyByTag('StudyDate', date_str)

    def getStudyByPID(self, PID):
        return self.getStudyByTag('PatientID', PID)

    def getStudyByTag(self, tagName, tagValue):
        for i in self:
            if i.getTag(tagName) == tagValue:
                return i

    def buildMSTable(self, DICOM_TAGS=SpydcmTK_config.SERIES_OVERVIEW_TAG_LIST):
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


def organiseDicoms(dcmDirectory, outputDirectory, anonName=None, anonID='', FORCE_READ=False, 
                   HIDE_PROGRESSBAR=False, REMOVE_PRIVATE_TAGS=False):
    """ Recurse down directory tree - grab dicoms and move to new
        hierarchical folder structure
    """
    if not HIDE_PROGRESSBAR:
        print('READING...')
    studies = ListOfDicomStudies.setFromInput(dcmDirectory, FORCE_READ=FORCE_READ, OVERVIEW=False, HIDE_PROGRESSBAR=HIDE_PROGRESSBAR)
    if not HIDE_PROGRESSBAR:
        print('WRITTING...')
    res = studies.writeToOrganisedFileStructure(outputDirectory, anonName=anonName, anonID=anonID, REMOVE_PRIVATE_TAGS=REMOVE_PRIVATE_TAGS)
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
def writeVTIToDicoms(vtiFile, dcmTemplateFile_or_ds, outputDir, arrayName=None, tagUpdateDict=None, patientMeta=None):
    if type(vtiFile) == str:
        vti = dcmVTKTK.fIO.readVTKFile(vtiFile)
        if vtiFile.endswith(".nii") or vtiFile.endswith(".nii.gz"):
            # TODO - need to also adjust patientMeta if this is nifti 
            vti.SetOrigin([i*0.001 for i in vti.GetOrigin()])
            vti.SetSpacing([i*0.001 for i in vti.GetSpacing()])
    else:
        vti = vtiFile
    if arrayName is None:
        A = dcmVTKTK.vtkfilters.getScalarsAsNumpy(vti)
    else:
        A = dcmVTKTK.vtkfilters.getArrayAsNumpy(vti, arrayName)
    if np.ndim(A) == 1:
        A = np.expand_dims(A, 1)
    dims = [0,0,0]
    vti.GetDimensions(dims)
    dims.append(A.shape[-1])
    A = np.reshape(A, dims, 'F')
    if patientMeta is None:
        patientMeta = dcmVTKTK.PatientMeta()
        patientMeta.initFromVTI(vti)
    return writeNumpyArrayToDicom(A, dcmTemplateFile_or_ds, patientMeta, outputDir, tagUpdateDict=tagUpdateDict)


def writeNumpyArrayToDicom(pixelArray, dcmTemplate_or_ds, patientMeta, outputDir, tagUpdateDict=None):
    if tagUpdateDict is None:
        tagUpdateDict = {}
    if dcmTemplate_or_ds is None:
        dsRAW = dcmTools.buildFakeDS()
    elif type(dcmTemplate_or_ds) == str:
        dsRAW = dicom.dcmread(dcmTemplate_or_ds)
    else:
        dsRAW = dcmTemplate_or_ds
    assert pixelArray.ndim == 4 and pixelArray.shape[3] in [1, 3, 4], "Input must be MxNxSx1, or MxNxSx3 or MxNxSx4 RGB(A) array"

    IS_RGB = False
    # Strip alpha if present
    if pixelArray.shape[3] == 4:
        pixelArray = pixelArray[:, :, :, :3]
        IS_RGB = True
    elif pixelArray.shape[3] == 3:
        IS_RGB = True

    # Ensure int16 
    NBIT = 16 
    if pixelArray.dtype != np.int16:
        print(f"WARNING: Converting pixelArray from {pixelArray.dtype} to int16")
        print(f"    PRE-CONVERSION:Max: {np.max(pixelArray)}, Min: {np.min(pixelArray)}")
        if np.max(pixelArray) <= 1.0:
            pixelArray = (pixelArray * 32767).astype(np.int16)
        else:
            pixelArray = np.clip(pixelArray, -32767, 32767)
            pixelArray = (pixelArray * 1.0).astype(np.int16)
        print(f"    POST-CONVERSION: Max: {np.max(pixelArray)}, Min: {np.min(pixelArray)}")

    nRow, nCol, nSlice, _ = pixelArray.shape
    mx, mn = np.max(pixelArray), 0
    try:
        slice0 = tagUpdateDict.pop('SliceLocation0')
    except KeyError:
        slice0 = patientMeta.SliceLocation0*dcmVTKTK.m_to_mm
    
    sliceThick = patientMeta.SliceThickness*dcmVTKTK.m_to_mm
    ipp = [i*dcmVTKTK.m_to_mm for i in patientMeta.ImagePositionPatient]

    SeriesUID = dicom.uid.generate_uid()
    try:
        SeriesNumber = tagUpdateDict.pop('SeriesNumber')
        try: # Have passed as Tag,VR,Value
            SeriesNumber = SeriesNumber[2]
        except TypeError:
            pass 
    except KeyError:
        try:
            SeriesNumber = dsRAW.SeriesNumber * 100
        except AttributeError:
            SeriesNumber = 99
    dsList = []
    dt = datetime.now()
    tags_to_copy = { # Tags to get from template and defaults if not found
        "PatientName": [0x00100010, "PN", "Anonymous^Name"],
        "PatientID": [0x00100020, "LO", "000000"],
        "Modality": [0x00080060, "CS", "OT"],
        "PatientBirthDate": [0x00100030, "DA",  ""],
        "PatientSex": [0x00100040, "CS", ""],
        "StudyDate": [0x00080020, "DA", dt.strftime('%Y%m%d')],
        "StudyTime": [0x00080030, "DM", dt.strftime('%H%M%S')],
        "InstitutionName": [0x00080080, "LO", ""],
        "StudyDescription": [0x00081030, "LO", "Study"],
        "SeriesDescription": [0x0008103e, "LO", "Image"],
        "StationName": [0x00081010, "SH", ""],
        "StudyInstanceUID": [0x0020000d, "UI", dicom.uid.generate_uid()],
        "StudyID": [0x00200010, "SH", None],
        "AccessionNumber": [0x00080050, "SH", None],
    }
    for k in range(nSlice):
        file_meta = dicom.Dataset()
        file_meta.MediaStorageSOPClassUID = dicom.uid.SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ##
        ds = dicom.FileDataset(filename_or_obj=None, dataset={}, file_meta=file_meta, preamble=b"\0" * 128,
                               is_implicit_VR=False, is_little_endian=True)
        # Copy tags from template DICOM
        for tag, default in tags_to_copy.items():
            iTag = dsRAW.get_item(tag)
            if iTag is not None:
                ds[tag] = iTag
            else:
                if default[2] is not None:
                    ds[tag] = dicom.DataElement(default[0], 
                                                default[1], 
                                                default[2])
        #
        # Set specific tags for this image conversion
        # If RGB then no position / orientation information
        ds.SeriesInstanceUID = SeriesUID
        ds.SOPInstanceUID = dicom.uid.generate_uid()
        #
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Rows = nRow
        ds.Columns = nCol
        ds.ImagesInAcquisition = nSlice
        ds.InStackPositionNumber = k+1
        ds.SeriesNumber = SeriesNumber
        ds.InstanceNumber = k+1
        if IS_RGB:
            ds.SamplesPerPixel = 3
            ds.PhotometricInterpretation = "RGB"
            ds.BitsAllocated = NBIT
            ds.BitsStored = NBIT
            ds.HighBit = NBIT - 1
            ds.PixelRepresentation = 0
            ds.PlanarConfiguration = 0  # Interleaved RGB
        else:
            ds.SamplesPerPixel = 1
            ds.BitsAllocated = NBIT
            ds.BitsStored = NBIT
            ds.HighBit = NBIT - 1
            if np.min(pixelArray) < 0:
                ds.PixelRepresentation = 1
            else:
                ds.PixelRepresentation = 0
            ds.PhotometricInterpretation = "MONOCHROME2"
        ds.SliceThickness = sliceThick # Can no longer claim overlapping slices if have modified
        ds.SpacingBetweenSlices = sliceThick
        ds.SmallestImagePixelValue = int(mn)
        ds.LargestImagePixelValue = int(mx)
        ds.WindowCenter = int(mx / 2)
        ds.WindowWidth = int(mx / 2)
        ds.PixelSpacing = [i*dcmVTKTK.m_to_mm for i in list(patientMeta.PixelSpacing)]
        #
        sliceVec = np.array(patientMeta.SliceVector)
        ImagePositionPatient = ipp + k*sliceVec*sliceThick
        ds.ImagePositionPatient = list(ImagePositionPatient)
        sliceLoc = slice0 + k*sliceThick
        ds.SliceLocation = sliceLoc
        ds.ImageOrientationPatient = list(patientMeta.ImageOrientationPatient)
        #
        # Set tags from method provided
        for iKey in tagUpdateDict.keys():
            if len(tagUpdateDict[iKey]) == 3: # Tag:0x00101010, VR, value
                ds.add_new(tagUpdateDict[iKey][0], tagUpdateDict[iKey][1], tagUpdateDict[iKey][2])
            else:
                ds[iKey] = tagUpdateDict[iKey]
        ##
        # Set PixelData
        ds.PixelData = pixelArray[:,:,k, :].tobytes()
        ds['PixelData'].VR = 'OW'
        dsList.append(ds)
    dcmSeries = DicomSeries(dsList, HIDE_PROGRESSBAR=True)
    return dcmSeries.writeToOrganisedFileStructure(outputDir)

def writeImageStackToDicom(images_sortedList, patientMeta, dcmTemplateFile_or_ds, 
                            outputDir, tagUpdateDict=None, CONVERT_TO_GREYSCALE=True):

    combinedImage = dcmVTKTK.readImageStackToVTI(images_sortedList, patientMeta, CONVERT_TO_GREYSCALE=CONVERT_TO_GREYSCALE)
    writeVTIToDicoms(combinedImage, 
                        dcmTemplateFile_or_ds=dcmTemplateFile_or_ds, 
                        outputDir=outputDir,
                        arrayName='PixelData',
                        tagUpdateDict=tagUpdateDict,
                        patientMeta=patientMeta)


def getResolution(dataVts):
    o,p1,p2,p3 = [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]
    i0,i1,j0,j1,k0,k1 = dataVts.GetExtent()
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

def _process_series_for_fdq(args):
    series, label, outDir = args
    series._loadToMemory()
    iOut = series.writeToVTS(os.path.join(outDir, f"{label}.pvd"))
    return (label, iOut)



