
from context import spydcmtk # type: ignore # This is useful for testing outside of environment

import os
import unittest
import shutil
from spydcmtk import dcmTK
from spydcmtk import spydcm
from spydcmtk.spydcm_config import SpydcmTK_config
import numpy as np


this_dir = os.path.split(os.path.realpath(__file__))[0]
TEST_DIRECTORY = os.path.join(this_dir, 'TEST_DATA')
TEST_DICOMS_DIR = os.path.join(TEST_DIRECTORY, 'DICOMS')
TEST_OUTPUT = os.path.join(this_dir, 'TEST_OUTPUT')
dcm001 = os.path.join(TEST_DICOMS_DIR, 'IM-00041-00001.dcm')
dcm00T = os.path.join(TEST_DICOMS_DIR, 'IM-00088-00001.dcm')
MISC_DIR = os.path.join(TEST_DIRECTORY, "MISC")
zipF = os.path.join(TEST_DIRECTORY, 'dicoms.zip')
vti001 = os.path.join(TEST_DIRECTORY, 'temp.vti')
image_npy = os.path.join(TEST_DIRECTORY, 'image.npy')
DEBUG = SpydcmTK_config.DEBUG
ThresL = 300
ThresH = 400

if DEBUG: 
    print('')
    print("WARNING - RUNNING IN DEBUG MODE - TEST OUTPUTS WILL NOT BE CLEANED")
    print('')

def cleanMakeDirs(idir):
    try:
        os.makedirs(idir)
    except FileExistsError:
        shutil.rmtree(idir)
        os.makedirs(idir)

class TestDicomSeries(unittest.TestCase):
    def runTest(self):
        listOfStudies = dcmTK.ListOfDicomStudies.setFromDirectory(TEST_DICOMS_DIR, HIDE_PROGRESSBAR=True)
        dcmStudy = listOfStudies.getStudyByDate('20140409')
        dcmSeries = dcmStudy.getSeriesBySeriesNumber(41)
        self.assertEqual(len(dcmSeries), 2, "Incorrect dicoms in dcmSeries")
        # ---
        self.assertEqual(dcmSeries.getNumberOfTimeSteps(), 2, msg="Incorrect read time steps")
        self.assertEqual(dcmSeries.getNumberOfSlicesPerVolume(), 1, msg="Incorrect read slices per vol")
        self.assertEqual(dcmSeries.getRootDir(), TEST_DICOMS_DIR, msg="Incorrect filename rootDir")
        self.assertEqual(dcmSeries.isCompressed(), False, msg="Incorrect compression read")
        self.assertEqual(dcmSeries.getSeriesNumber(), 41, msg="Incorrect series number")
        self.assertEqual(dcmSeries.getSeriesOutDirName(), 'SE41_Cine_TruFisp_RVLA', msg="Incorrect series directory save name")
        self.assertEqual(dcmSeries.getTag('PatientName'), 'ANON', msg="Incorrect TAG name")
        self.assertAlmostEqual(dcmSeries.sliceLocations[1], -26.291732075, places=7, msg='Slice location incorrect')
        self.assertAlmostEqual(dcmSeries.getDeltaRow(), 1.875, places=7, msg='deltaRow incorrect')
        self.assertAlmostEqual(dcmSeries.getDeltaCol(), 1.875, places=7, msg='deltaCol incorrect')
        self.assertAlmostEqual(dcmSeries.getTemporalResolution(), 51.92, places=7, msg='deltaTime incorrect')
        self.assertEqual(dcmSeries.IS_SIEMENS(), True, msg="Incorrect manufacturer")
        self.assertEqual(dcmSeries.getPulseSequenceName(), '*tfi2d1_12', msg="Incorrect sequence name")
        tmpDir = os.path.join(TEST_OUTPUT, 'tmp1')
        cleanMakeDirs(tmpDir)
        dcmSeries.writeToOrganisedFileStructure(tmpDir)
        dcmStudy.anonymise(anonName='Not A Name', anonPatientID='12345')
        dcmSeries.writeToOrganisedFileStructure(tmpDir)
        if not DEBUG:
            shutil.rmtree(tmpDir)
        


class TestDicomStudy(unittest.TestCase):
    def runTest(self):
        listOfStudies = dcmTK.ListOfDicomStudies.setFromDirectory(TEST_DICOMS_DIR, HIDE_PROGRESSBAR=True)
        dcmStudy = listOfStudies.getStudyByDate('20140409')
        self.assertEqual(len(dcmStudy), 1, "Incorrect number series in dcmStudy")
        patOverview = dcmStudy.getPatientOverview()
        self.assertEqual(patOverview[0][4], "PatientAge", "Patient overview incorrect")
        self.assertEqual(patOverview[1][4], "033Y", "Patient overview incorrect")
        studyOverview = dcmStudy.getStudyOverview()
        self.assertEqual(studyOverview[0][4], "StudyDate", "Study overview incorrect")
        self.assertEqual(studyOverview[1][4], "20140409", "Study overview incorrect")
        seriesOverview = dcmStudy[0].getSeriesOverview()
        self.assertEqual(seriesOverview[0][1], "SeriesDescription", "Series overview incorrect")
        self.assertEqual(seriesOverview[1][1], "Cine_TruFisp_RVLA", "Series overview incorrect")
        tmpDir = os.path.join(TEST_OUTPUT, 'tmp2')
        cleanMakeDirs(tmpDir)
        dcmStudy.writeToOrganisedFileStructure(tmpDir)
        dcmStudy.anonymise(anonName='Not A Name', anonPatientID='')
        dcmStudy.resetUIDs()
        dcmStudy.writeToOrganisedFileStructure(tmpDir)
        if not DEBUG:
            shutil.rmtree(tmpDir)


class TestDicom2VTK(unittest.TestCase):
    def runTest(self):
        listOfStudies = dcmTK.ListOfDicomStudies.setFromDirectory(TEST_DICOMS_DIR, HIDE_PROGRESSBAR=True)
        for dcmStudy in listOfStudies:
            dcmSeries = dcmStudy.getSeriesBySeriesNumber(41)
            if dcmSeries is not None:
                break
        vtiDict = dcmSeries.buildVTIDict()
        self.assertAlmostEqual(list(vtiDict.keys())[1], 0.05192, places=7, msg='time key in vti dict incorrect')
        tmpDir = os.path.join(TEST_OUTPUT, 'tmp3')
        cleanMakeDirs(tmpDir)
        fOut = dcmSeries.writeToVTI(tmpDir)
        self.assertTrue(os.path.isfile(fOut), msg='Written pvd file does not exist')
        dd = dcmTK.dcmVTKTK.fIO.readPVD(fOut)
        dTimes = sorted(dd.keys())
        self.assertAlmostEqual(dTimes[1], 0.05192, places=7, msg='time key in vti dict incorrect')
        vti0 = dd[dTimes[0]]
        oo = vti0.GetOrigin()
        self.assertAlmostEqual(oo[1], 0.1166883275304, places=7, msg='origin in vti dict incorrect')
        if not DEBUG:
            dcmTK.dcmVTKTK.fIO.deleteFilesByPVD(fOut)
            shutil.rmtree(tmpDir)


class TestDicom2MSTable(unittest.TestCase):
    def runTest(self):
        tmpDir = os.path.join(TEST_OUTPUT, 'tmp4')
        cleanMakeDirs(tmpDir)
        fOut = spydcm.buildTableOfDicomParamsForManuscript([TEST_DICOMS_DIR], 
                                                           outputCSVPath=os.path.join(tmpDir, 'ms.csv'), 
                                                           seriesDescriptionIdentifier='RVLA',
                                                           ONE_FILE_PER_DIR=False)
        self.assertTrue(os.path.isfile(fOut), msg='Written MS csv file does not exist')
        if not DEBUG:
            shutil.rmtree(tmpDir)

class TestDicomPixDataArray(unittest.TestCase):
    def runTest(self):
        listOfStudies = dcmTK.ListOfDicomStudies.setFromDirectory(TEST_DICOMS_DIR, HIDE_PROGRESSBAR=True)
        dcmStudy = listOfStudies.getStudyByTag('StudyInstanceUID', '1.2.826.0.1.3680043.8.498.46701999696935009211199968005189443301')
        dcmSeries = dcmStudy.getSeriesBySeriesNumber(99)
        A, patientMeta = dcmSeries.getPixelDataAsNumpy()
        self.assertEqual(A[17,13,0], 1935, msg='Pixel1 data not matching expected') 
        self.assertEqual(A[17,13,1], 2168, msg='Pixel2 data not matching expected') 
        self.assertEqual(A[17,13,2], 1773, msg='Pixel3 data not matching expected') 
        self.assertEqual(patientMeta.Origin[2], 0.0003, msg='Origin data not matching expected') 
        # if DEBUG:
        #     import matplotlib.pyplot as plt
        #     for k1 in range(A.shape[-1]):
        #         for k2 in range(A.shape[-2]):
        #             plt.imshow(A[:,:,k2, k1])
        #             plt.show()

class TestDicomPixDataMeta(unittest.TestCase):

    def runTest(self):
        listOfStudies = dcmTK.ListOfDicomStudies.setFromDirectory(TEST_DICOMS_DIR, HIDE_PROGRESSBAR=True)
        dcmStudy = listOfStudies.getStudyByTag('StationName', 'AWP45557')
        dcmSeries = dcmStudy.getSeriesBySeriesNumber(41)
        A, patientMeta = dcmSeries.getPixelDataAsNumpy()
        self.assertEqual(patientMeta.Times[1], 0.05192, msg='Time data not matching expected') 
        self.assertAlmostEqual(patientMeta.Origin[1], 0.11668832753047001, msg='Origin data not matching expected') 
        self.assertAlmostEqual(patientMeta.ImageOrientationPatient[1], -0.540900243742, msg='ImageOrientationPatient data not matching expected') 
        self.assertEqual(patientMeta.PatientPosition, 'HFS', msg='PatientPosition data not matching expected') 
        # if DEBUG:
        #     import matplotlib.pyplot as plt
        #     for k1 in range(A.shape[-1]):
        #         for k2 in range(A.shape[-2]):
        #             plt.imshow(A[:,:,k2, k1])
        #             plt.show()

class TestDicom2HTML(unittest.TestCase):
    def runTest(self):
        tmpDir = os.path.join(TEST_OUTPUT, 'tmp5')
        cleanMakeDirs(tmpDir)
        fOut = spydcm.convertInputsToHTML([vti001], tmpDir, QUIET=True)
        self.assertTrue(os.path.isfile(fOut), msg='Written html file does not exist')
        if not DEBUG:
            shutil.rmtree(tmpDir)

class TestDicom2VTI(unittest.TestCase):
    def runTest(self):
        tmpDir = os.path.join(TEST_OUTPUT, 'tmpDCM2VTI')
        cleanMakeDirs(tmpDir)
        vtiOutA = os.path.join(tmpDir, 'A.vti')
        vtiOutB = os.path.join(tmpDir, 'B.vti')
        fOut = spydcm.directoryToVTI(MISC_DIR, vtiOutA)
        fOut = spydcm.directoryToVTI(MISC_DIR, vtiOutB, TRUE_ORIENTATION=True)
        self.assertTrue(os.path.isfile(vtiOutA), msg='Written vtiA file does not exist')
        self.assertTrue(os.path.isfile(vtiOutB), msg='Written vtiB file does not exist')
        if not DEBUG:
            shutil.rmtree(tmpDir)

class TestStream(unittest.TestCase):
    def runTest(self):
        tmpDir = os.path.join(TEST_OUTPUT, 'tmp6')
        cleanMakeDirs(tmpDir)
        spydcm.dcmTools.streamDicoms(TEST_DICOMS_DIR, tmpDir, FORCE_READ=False, HIDE_PROGRESSBAR=True, SAFE_NAMING=False)
        expectedOutput = os.path.join(this_dir, "TEST_OUTPUT/tmp6/TEST-DATA_12345/20000101_1088/SE88_SeriesLaugh/IM-00088-00001.dcm")
        self.assertTrue(os.path.isfile(expectedOutput), msg='Stream failed')
        if not DEBUG:
            shutil.rmtree(tmpDir)
        ## Test with safe
        tmpDir = os.path.join(TEST_OUTPUT, 'tmp7')
        cleanMakeDirs(tmpDir)
        spydcm.dcmTools.streamDicoms(TEST_DICOMS_DIR, tmpDir, FORCE_READ=False, HIDE_PROGRESSBAR=True, SAFE_NAMING=True)
        expectedOutput = os.path.join(this_dir, "TEST_OUTPUT/tmp7/ANON/1.3.12.2.1107.5.2.19.45557.30000014040822145264600000001/1.3.12.2.1107.5.2.19.45557.2014040909463893489380900.0.0.0/IM-1.3.12.2.1107.5.2.19.45557.2014040909463913941980942.dcm")
        self.assertTrue(os.path.isfile(expectedOutput), msg='Stream failed (SAFE)')
        if not DEBUG:
            shutil.rmtree(tmpDir)


class TestZipAndUnZip(unittest.TestCase):
    def runTest(self):
        if not os.path.isfile(zipF):
            print(f"WARNING: UnZip test not run - {zipF} not found")
            return # Don't have data - can not run test
        tmpDir = os.path.join(TEST_OUTPUT, 'tmpzip')
        LDS = dcmTK.ListOfDicomStudies.setFromInput(zipF)
        studyA = LDS.getStudyByTag("StudyID", "1088")
        studyB = LDS.getStudyByTag("StudyInstanceUID", "1.2.826.0.1.3680043.8.498.46701999696935009211199968005189443301")
        self.assertTrue(len(studyA.getSeriesBySeriesNumber(88))==3, msg='Incorrect number of images for series 88')
        self.assertTrue(len(studyB.getSeriesBySeriesNumber(99))==3, msg='Incorrect number of images for series 99')
        outputs = LDS.writeToZipArchive(tmpDir, CLEAN_UP=False)
        resFileA = os.path.join(tmpDir, "TEST-DATA_1088_20000101.zip")
        self.assertTrue(os.path.isfile(resFileA), msg='Written zip file does not exist')
        self.assertTrue(os.path.isdir(resFileA[:-4]), msg='Written zip temp directory does not exist')
        shutil.rmtree(resFileA[:-4])
        os.unlink(resFileA)
        outputs = LDS.writeToZipArchive(tmpDir, CLEAN_UP=True)
        self.assertTrue(os.path.isfile(resFileA), msg='Written zip file does not exist')
        self.assertFalse(os.path.isdir(resFileA[:-4]), msg='Written zip temp directory does exist - should have been cleaned up')
        if not DEBUG:
            shutil.rmtree(tmpDir)


class TestImageToDicom(unittest.TestCase):
    def runTest(self):
        pixArray = np.load(image_npy)
        tmpDir = os.path.join(TEST_OUTPUT, 'tmp8')
        cleanMakeDirs(tmpDir)
        patMatrix = {'PixelSpacing': [0.02, 0.02], 
                     'ImagePositionPatient': [0.0, 0.1, 0.3], 
                     'ImageOrientationPatient': [0.0,0.0,1.0,0.0,1.0,0.0], 
                     'SliceThickness': 0.04,
                     'SpacingBetweenSlices': 0.04}
        patMeta = dcmTK.dcmVTKTK.PatientMeta()
        patMeta.initFromDictionary(patMatrix)
        tagUpdateDict = {'SeriesNumber': 99, 
                         'StudyDescription': ([0x0008,0x1030], 'LO', "TestDataA"), 
                         'SeriesDescription': ([0x0008,0x103e], 'LO', "SeriesWink"), 
                         'StudyID': ([0x0020,0x0010], 'SH', '1099')}
        # pixArray must by nCxnRxnSxnChannels
        pixArray = np.expand_dims(pixArray, 3)
        dcmTK.writeNumpyArrayToDicom(pixArray[:,:,:3,:], None, patMeta, tmpDir)
        if not DEBUG:
            shutil.rmtree(tmpDir)

        pixArray = np.load(image_npy)
        tmpDir = os.path.join(TEST_OUTPUT, 'tmp9')
        cleanMakeDirs(tmpDir)
        patMatrix = {'PixelSpacing': [0.02, 0.02], 
                     'ImagePositionPatient': [0.0, 0.1, 0.3], 
                     'ImageOrientationPatient': [0.0,0.0,1.0,0.0,1.0,0.0], 
                     'SliceThickness': 0.04,
                     'SpacingBetweenSlices': 0.04}
        patMeta = dcmTK.dcmVTKTK.PatientMeta()
        patMeta.initFromDictionary(patMatrix)
        tagUpdateDict = {'SeriesNumber': 88, 
                         'StudyDescription': ([0x0008,0x1030], 'LO', "TestDataB"), 
                         'SeriesDescription': ([0x0008,0x103e], 'LO', "SeriesLaugh"), 
                         'StudyID': ([0x0020,0x0010], 'SH', '1088')}
        # pixArray must by nCxnRxnSxnChannels
        pixArray = np.expand_dims(pixArray, 3)
        dcmTK.writeNumpyArrayToDicom(pixArray[:,:,3:,:], None, patMeta, tmpDir, tagUpdateDict=tagUpdateDict)
        if not DEBUG:
            shutil.rmtree(tmpDir)

def getTestVolDS():
    dsList = []
    if os.path.isdir(MISC_DIR):
        dsList = dcmTK.DicomSeries.setFromDirectory(MISC_DIR, HIDE_PROGRESSBAR=True)
    return dsList

# class TestArrToDCMSeg(unittest.TestCase):
#     def runTest(self):
#         dsList = getTestVolDS()
#         if len(dsList) > 0:
#             tmpDir = os.path.join(TEST_OUTPUT, 'tmp10')
#             cleanMakeDirs(tmpDir)
#             pixArray = np.transpose(np.squeeze(dsList.getPixelDataAsNumpy()[0]), axes=[2,0,1])
#             dcmSegOut = os.path.join(tmpDir, 'dcmseg.dcm')
#             lHigh = pixArray > ThresL
#             lLow = pixArray < ThresH
#             labelMap = np.array((lLow.astype(int)+lHigh.astype(int))==2).astype(int)
#             dcmTK.dcmVTKTK.array_to_DcmSeg(labelMap, dsList, dcmSegOut)
#             self.assertTrue(os.path.isfile(dcmSegOut), msg='DCMSEG file does not exist')
#             if not DEBUG:
#                 shutil.rmtree(tmpDir)

def _scaleVTI(ff):
    ii = dcmTK.dcmVTKTK.fIO.readVTKFile(ff)
    dcmTK.dcmVTKTK.scaleVTI(ii, 1000.0)
    oo = ii.GetOrigin()
    dcmTK.dcmVTKTK.fIO.writeVTKFile(ii, ff)
    return oo

# class TestDCMSegToVTI(unittest.TestCase):
#     def runTest(self):
#         dsList = getTestVolDS()
#         if len(dsList) > 0:
#             tmpDir = os.path.join(TEST_OUTPUT, 'tmp11')
#             cleanMakeDirs(tmpDir)
#             pixArray = np.transpose(np.squeeze(dsList.getPixelDataAsNumpy()[0]), axes=[2,0,1])
#             vtiOutA = os.path.join(tmpDir, 'dcm.vti')
#             vtiOutA2 = os.path.join(tmpDir, 'dcm_TRUE_ORIENTATION.vti')
#             vtsA = os.path.join(tmpDir, 'dcm_TRUE_ORIENTATION_VTS.vts')
#             dsList.writeToVTI(vtiOutA, TRUE_ORIENTATION=False)
#             dsList.writeToVTS(vtsA)
#             dsList.writeToVTI(vtiOutA2, TRUE_ORIENTATION=True)

#             vtk_DcmSegOutA = os.path.join(tmpDir, 'dcmSeg.vts')
#             lHigh = pixArray > ThresL
#             lLow = pixArray < ThresH
#             labelMap = np.array((lLow.astype(int)+lHigh.astype(int))==2).astype(int)
#             dcmSegOut = os.path.join(tmpDir, 'dcmseg.dcm')
#             dcmTK.dcmVTKTK.array_to_DcmSeg(labelMap, dsList, dcmSegOut)
#             dcmTK.dcmVTKTK.dicom_seg_to_vtk(dcmSegOut, vtk_DcmSegOutA, TRUE_ORIENTATION=True)

#             vtk_DcmSegOutB = os.path.join(tmpDir, 'dcmSegB.vts')
#             dcmTK.dcmVTKTK.dicom_seg_to_vtk(os.path.join(TEST_DIRECTORY, "example_seg.dcm"), vtk_DcmSegOutB, TRUE_ORIENTATION=True)

#             self.assertTrue(os.path.isfile(vtiOutA), msg='VTI from DCMSEG file does not exist')
#             self.assertTrue(os.path.isfile(vtk_DcmSegOutA), msg='VTI-SEG-A from DCMSEG file does not exist')
#             self.assertTrue(os.path.isfile(vtk_DcmSegOutB), msg='VTI-SEG-B from DCMSEG file does not exist')
#             if not DEBUG:
#                 shutil.rmtree(tmpDir)

def buildImages(outDir, seNum):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"WARNING: no matplotlib - Images not built")
        return False
    filelist = [os.path.join(TEST_DICOMS_DIR, i) for i in os.listdir(TEST_DICOMS_DIR) if i.startswith(f"IM-{seNum:05d}")]
    dsSeries = spydcm.dcmTK.DicomSeries.setFromFileList(filelist, HIDE_PROGRESSBAR=True)
    arr, _ = dsSeries.getPixelDataAsNumpy()
    m,n,o,_ = arr.shape
    for k1 in range(o):
        arr2D = arr[:,:,k1]
        fig, axs = plt.subplots(1,1)
        axs.imshow(arr2D, cmap='gray')
        axs.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(os.path.join(outDir, f"IM-{k1:04d}.jpg"), bbox_inches='tight', pad_inches=0)
        plt.close()
    return True

class TestImagesToVTI(unittest.TestCase):
    def runTest(self):
        tmpDir = os.path.join(TEST_OUTPUT, 'tmpImg2VTI')
        cleanMakeDirs(tmpDir)
        res = buildImages(tmpDir, 99)
        if res:
            fileList = [os.path.join(tmpDir, i) for i in os.listdir(tmpDir) if i.endswith('jpg')]
            fileList = sorted(fileList)
            if len(fileList) > 0:
                pat_meta = dcmTK.dcmVTKTK.PatientMeta()
                pat_meta.initFromDictionary({'Origin': [0.0,0.0,0.0],
                                             'Spacing': [0.001, 0.001, 0.02]})
                ii = dcmTK.dcmVTKTK.readImageStackToVTI(fileList, patientMeta=pat_meta, CONVERT_TO_GREYSCALE=True)
                emojivti = os.path.join(tmpDir, 'emoji.vti')
                dcmTK.dcmVTKTK.fIO.writeVTKFile(ii, emojivti)
                self.assertTrue(os.path.isfile(emojivti), msg='emoji.vti file does not exist')
                valA = ii.GetPointData().GetArray("PixelData").GetTuple(401239)[0]
                IDB = ii.ComputePointId([440,355,1])
                valB = ii.GetPointData().GetArray("PixelData").GetTuple(IDB)[0]
                self.assertEqual(valA, 54, "Image to VTI data incorrect")
                self.assertEqual(valB, 28, "Image to VTI data incorrect")
                #
                ii2 = dcmTK.dcmVTKTK.readImageStackToVTI(fileList, patientMeta=None, CONVERT_TO_GREYSCALE=False)
                emojivti2 = os.path.join(tmpDir, 'emoji2.vti')
                dcmTK.dcmVTKTK.fIO.writeVTKFile(ii2, emojivti2)
                self.assertTrue(os.path.isfile(emojivti2), msg='emoji2.vti file does not exist')
                valA = ii2.GetPointData().GetArray("PixelData").GetTuple(89786)[2]
                IDB = ii2.ComputePointId([27,187,0])
                valB = ii2.GetPointData().GetArray("PixelData").GetTuple(IDB)[2]
                self.assertEqual(valA, 113, "Image to VTI data incorrect")
                self.assertEqual(valB, 253, "Image to VTI data incorrect")
        if not DEBUG:
            shutil.rmtree(tmpDir)

class TestImagesToDCM(unittest.TestCase):
    def runTest(self):
        tmpDir = os.path.join(TEST_OUTPUT, 'tmpImg2DCM')
        cleanMakeDirs(tmpDir)
        res = buildImages(tmpDir, 88)
        if res:
            fileList = [os.path.join(tmpDir, i) for i in os.listdir(tmpDir) if i.endswith('jpg')]
            fileList = sorted(fileList)
            if len(fileList) > 0:
                pat_meta = dcmTK.dcmVTKTK.PatientMeta()
                pat_meta.initFromDictionary({'Origin': [0.0,0.0,0.0],
                                             'Spacing': [0.001, 0.001, 0.02],
                                             'ImageOrientationPatient': [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]})
                dcmTK.writeImageStackToDicom(fileList, patientMeta=pat_meta, dcmTemplateFile_or_ds=dcm00T,
                                                outputDir=tmpDir)
                imageDS = dcmTK.DicomSeries.setFromDirectory(tmpDir, HIDE_PROGRESSBAR=True)
                arr, _ = imageDS.getPixelDataAsNumpy()
                self.assertEqual(arr[179,153,0,0], 163, "Dicom orientation for image2DCM wrong")
        if not DEBUG:
            shutil.rmtree(tmpDir)


class TestDCM2VTI2DCM(unittest.TestCase):
    def runTest(self):
        dcmSeries = getTestVolDS()
        A, meta = dcmSeries.getPixelDataAsNumpy()
        Ashape, A_id = A.shape[:3], A[179,153,44,0]
        if len(dcmSeries) > 0:
            tmpDir = os.path.join(TEST_OUTPUT, 'tmpDCM2VTI2DCM')
            cleanMakeDirs(tmpDir)
            vtiOut = os.path.join(tmpDir, 'dcm.vti')
            dcmSeries.writeToVTI(vtiOut)
            vtiObj = dcmTK.dcmVTKTK.fIO.readVTKFile(vtiOut)
            A2 = dcmTK.dcmVTKTK.vtkfilters.getArrayAsNumpy(vtiObj, "PixelData", RETURN_3D=True)
            A2shape, A2_id = A2.shape, A2[179,153,44]
            self.assertEqual(A_id, A2_id, "Pixel data incorrect")
            self.assertEqual(Ashape, A2shape, "Array shape incorrect")
            vtiObj_m = dcmTK.dcmVTKTK.vtkfilters.filterVtiMedian(vtiObj, filterKernalSize=3)
            newDcmDir = dcmTK.writeVTIToDicoms(vtiObj, dcmSeries[0], tmpDir)
            dcmSeries2 = dcmTK.DicomSeries.setFromDirectory(newDcmDir, HIDE_PROGRESSBAR=True)
            A3, meta2 = dcmSeries2.getPixelDataAsNumpy()
            A3shape, A3_id = A3.shape[:3], A3[179,153,44]
            self.assertEqual(A3_id, A_id, "Pixel data incorrect")
            self.assertEqual(A3shape, Ashape, "Array shape incorrect")
            if not DEBUG:
                shutil.rmtree(tmpDir)



class Test_SetTagValues(unittest.TestCase):
    def runTest(self):
        # METHOD A
        listOfStudies = dcmTK.ListOfDicomStudies.setFromDirectory(TEST_DICOMS_DIR, HIDE_PROGRESSBAR=True)
        dcmStudy = listOfStudies.getStudyByDate('20140409')
        dcmStudy.setTags_all(0x00080020, "19901231") # change date of study
        tmpDir = os.path.join(TEST_OUTPUT, 'tmpSetTags')
        cleanMakeDirs(tmpDir)
        listOfStudies.writeToOrganisedFileStructure(tmpDir)

        listOfStudies2 = dcmTK.ListOfDicomStudies.setFromDirectory(tmpDir, HIDE_PROGRESSBAR=True)
        dcmStudy2 = listOfStudies2.getStudyByDate('19901231')
        self.assertEqual(len(dcmStudy2), 1, "Incorrect number series in dcmStudy")
        
        # METHOD B
        listOfStudies = dcmTK.ListOfDicomStudies.setFromDirectory(TEST_DICOMS_DIR, HIDE_PROGRESSBAR=True)
        dcmStudy = listOfStudies.getStudyByDate('20140409')
        dcmStudy.setTags_all("StudyDate", "19901231") # change date of study
        tmpDir = os.path.join(TEST_OUTPUT, 'tmpSetTags')
        cleanMakeDirs(tmpDir)
        listOfStudies.writeToOrganisedFileStructure(tmpDir)

        listOfStudies2 = dcmTK.ListOfDicomStudies.setFromDirectory(tmpDir, HIDE_PROGRESSBAR=True)
        dcmStudy2 = listOfStudies2.getStudyByDate('19901231')
        self.assertEqual(len(dcmStudy2), 1, "Incorrect number series in dcmStudy")
        
        if not DEBUG:
            shutil.rmtree(tmpDir)



if __name__ == '__main__':
    unittest.main()

    # DEBUG = True
    # suite = unittest.TestSuite()
    # suite.addTest(TestImagesToDCM('runTest'))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)