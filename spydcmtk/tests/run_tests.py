
from context import spydcmtk # This is useful for testing outside of environment

import os
import unittest
import shutil
from spydcmtk import dcmTK
from spydcmtk import spydcm
from spydcmtk.spydcm_config import SpydcmTK_config
import numpy as np


this_dir = os.path.split(os.path.realpath(__file__))[0]
TEST_DIRECTORY = os.path.join(this_dir, 'TEST_DATA')
TEST_OUTPUT = os.path.join(this_dir, 'TEST_OUTPUT')
dcm001 = os.path.join(TEST_DIRECTORY, 'IM-00041-00001.dcm')
vti001 = os.path.join(TEST_DIRECTORY, 'temp.vti')
imnpy = os.path.join(TEST_DIRECTORY, 'image.npy')
DEBUG = SpydcmTK_config.DEBUG

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
        listOfStudies = dcmTK.ListOfDicomStudies.setFromDirectory(TEST_DIRECTORY, HIDE_PROGRESSBAR=True)
        dcmStudy = listOfStudies.getStudyByDate('20140409')
        dcmSeries = dcmStudy.getSeriesBySeriesNumber(41)
        self.assertEqual(len(dcmSeries), 2, "Incorrect dicoms in dcmSeries")
        # ---
        self.assertEqual(dcmSeries.getNumberOfTimeSteps(), 2, msg="Incorrect read time steps")
        self.assertEqual(dcmSeries.getNumberOfSlicesPerVolume(), 1, msg="Incorrect read slices per vol")
        self.assertEqual(dcmSeries.getRootDir(), TEST_DIRECTORY, msg="Incorrect filename rootDir")
        self.assertEqual(dcmSeries.isCompressed(), False, msg="Incorrect compression read")
        self.assertEqual(dcmSeries.getSeriesNumber(), 41, msg="Incorrect series number")
        self.assertEqual(dcmSeries.getSeriesOutDirName(), 'SE41_Cine_TruFisp_RVLA', msg="Incorrect series directory save name")
        self.assertEqual(dcmSeries.getTag('PatientName'), 'ANON', msg="Incorrect TAG name")
        self.assertAlmostEqual(dcmSeries.sliceLocations[1], -26.291732075, places=7, msg='Slice location incorrect')
        self.assertAlmostEqual(dcmSeries.getDeltaRow(), 1.875, places=7, msg='deltaRow incorrect')
        self.assertAlmostEqual(dcmSeries.getDeltaCol(), 1.875, places=7, msg='deltaCol incorrect')
        self.assertAlmostEqual(dcmSeries.getTemporalResolution(), 51.92, places=7, msg='deltaTime incorrect')
        self.assertEqual(dcmSeries.IS_SIEMENS(), True, msg="Incorrect manufactuer")
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
        listOfStudies = dcmTK.ListOfDicomStudies.setFromDirectory(TEST_DIRECTORY, HIDE_PROGRESSBAR=True)
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


class TestDicom2VT2Dicom(unittest.TestCase):
    def runTest(self):
        listOfStudies = dcmTK.ListOfDicomStudies.setFromDirectory(TEST_DIRECTORY, HIDE_PROGRESSBAR=True)
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
        dd = dcmTK.dcmVTKTK.readPVD(fOut)
        dTimes = sorted(dd.keys())
        self.assertAlmostEqual(dTimes[1], 0.05192, places=7, msg='time key in vti dict incorrect')
        vti0 = dd[dTimes[0]]
        oo = vti0.GetOrigin()
        self.assertAlmostEqual(oo[1], 0.1166883275304, places=7, msg='origin in vti dict incorrect')
        if not DEBUG:
            dcmTK.dcmVTKTK.deleteFilesByPVD(fOut)
            shutil.rmtree(tmpDir)


class TestDicom2MSTable(unittest.TestCase):
    def runTest(self):
        tmpDir = os.path.join(TEST_OUTPUT, 'tmp4')
        cleanMakeDirs(tmpDir)
        fOut = spydcm.buildTableOfDicomParamsForManuscript([TEST_DIRECTORY], 
                                                           outputCSVPath=os.path.join(tmpDir, 'ms.csv'), 
                                                           seriesDescriptionIdentifier='RVLA',
                                                           ONE_FILE_PER_DIR=False)
        self.assertTrue(os.path.isfile(fOut), msg='Written MS csv file does not exist')
        if not DEBUG:
            shutil.rmtree(tmpDir)

class TestDicomPixDataArray(unittest.TestCase):
    def runTest(self):
        listOfStudies = dcmTK.ListOfDicomStudies.setFromDirectory(TEST_DIRECTORY, HIDE_PROGRESSBAR=True)
        dcmStudy = listOfStudies.getStudyByTag('StudyInstanceUID', '1.2.826.0.1.3680043.8.498.46701999696935009211199968005189443301')
        dcmSeries = dcmStudy.getSeriesBySeriesNumber(99)
        A, meta = dcmSeries.getPixelDataAsNumpy()
        self.assertEquals(A[17,13,0], 1935, msg='Pixel1 data not matching expected') 
        self.assertEquals(A[17,13,1], 2168, msg='Pixel2 data not matching expected') 
        self.assertEquals(A[17,13,2], 1773, msg='Pixel3 data not matching expected') 
        self.assertEquals(meta['Origin'][2], 0.0003, msg='Origin data not matching expected') 
        if DEBUG:
            import matplotlib.pyplot as plt
            for k1 in range(A.shape[-1]):
                for k2 in range(A.shape[-2]):
                    plt.imshow(A[:,:,k2, k1])
                    plt.show()

class TestDicomPixDataMeta(unittest.TestCase):

    def runTest(self):
        listOfStudies = dcmTK.ListOfDicomStudies.setFromDirectory(TEST_DIRECTORY, HIDE_PROGRESSBAR=True)
        dcmStudy = listOfStudies.getStudyByTag('StationName', 'AWP45557')
        dcmSeries = dcmStudy.getSeriesBySeriesNumber(41)
        A, meta = dcmSeries.getPixelDataAsNumpy()
        print(A.shape)
        print(meta)
        self.assertEquals(meta['Times'][1], 0.05192, msg='Time data not matching expected') 
        self.assertAlmostEquals(meta['Origin'][1], 0.11668832753047001, msg='Origin data not matching expected') 
        self.assertAlmostEquals(meta['ImageOrientationPatient'][1], -0.540900243742, msg='ImageOrientationPatient data not matching expected') 
        self.assertEquals(meta['PatientPosition'], 'HFS', msg='PatientPosition data not matching expected') 
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
        tmpDir = os.path.join(TEST_OUTPUT, 'tmpDCM2VIT')
        cleanMakeDirs(tmpDir)
        fOut = spydcm.directoryToVTI(dcm001, tmpDir)
        for iFOut in fOut:
            self.assertTrue(os.path.isfile(iFOut), msg='Written vti file does not exist')
        if not DEBUG:
            shutil.rmtree(tmpDir)

class TestStream(unittest.TestCase):
    def runTest(self):
        tmpDir = os.path.join(TEST_OUTPUT, 'tmp6')
        cleanMakeDirs(tmpDir)
        spydcm.dcmTools.streamDicoms(TEST_DIRECTORY, tmpDir, FORCE_READ=False, HIDE_PROGRESSBAR=True, SAFE_NAMING=False)
        expectedOutput = os.path.join(this_dir, "TEST_OUTPUT/tmp6/TEST-DATA_12345/20000101_1088/SE88_SeriesLaugh/IM-00088-00001.dcm")
        self.assertTrue(os.path.isfile(expectedOutput), msg='Stream failed')
        if not DEBUG:
            shutil.rmtree(tmpDir)
        ## Test with safe
        tmpDir = os.path.join(TEST_OUTPUT, 'tmp7')
        cleanMakeDirs(tmpDir)
        spydcm.dcmTools.streamDicoms(TEST_DIRECTORY, tmpDir, FORCE_READ=False, HIDE_PROGRESSBAR=True, SAFE_NAMING=True)
        expectedOutput = os.path.join(this_dir, "TEST_OUTPUT/tmp7/ANON/1.3.12.2.1107.5.2.19.45557.30000014040822145264600000001/1.3.12.2.1107.5.2.19.45557.2014040909463893489380900.0.0.0/IM-1.3.12.2.1107.5.2.19.45557.2014040909463913941980942.dcm")
        self.assertTrue(os.path.isfile(expectedOutput), msg='Stream failed (SAFE)')
        if not DEBUG:
            shutil.rmtree(tmpDir)


class TestZipAndUnZip(unittest.TestCase):
    def runTest(self):
        zipF = "/Volume/TEST/zipped.zip"
        if not os.path.isfile(zipF):
            print(f"WARNING: UnZip test not run - {zipF} not found")
            return # Don't have data - can not run test
        LDS = dcmTK.ListOfDicomStudies.setFromInput(zipF)
        self.assertTrue(len(LDS[0].getSeriesBySeriesNumber(8))==5, msg='Incorrect number of images for series 8')
        self.assertTrue(len(LDS[0].getSeriesBySeriesNumber(9))==6, msg='Incorrect number of images for series 9')
        
        tempTestDir = os.path.split(zipF)[0]
        outputs = LDS.writeToZipArchive(tempTestDir, CLEAN_UP=False)
        self.assertTrue(os.path.isfile(outputs[0]), msg='Written zip file does not exist')
        self.assertTrue(os.path.isdir(outputs[0][:-4]), msg='Written zip temp directory does not exist')
        shutil.rmtree(outputs[0][:-4])
        os.unlink(outputs[0])
        outputs = LDS.writeToZipArchive(tempTestDir, CLEAN_UP=True)
        self.assertTrue(os.path.isfile(outputs[0]), msg='Written zip file does not exist')
        self.assertFalse(os.path.isdir(outputs[0][:-4]), msg='Written zip temp directory does exist - should have been cleaned up')
        os.unlink(outputs[0])


class TestImageToDicom(unittest.TestCase):
    def runTest(self):
        pixArray = np.load(imnpy)
        tmpDir = os.path.join(TEST_OUTPUT, 'tmp8')
        cleanMakeDirs(tmpDir)
        patMatrix = {'PixelSpacing': [0.02, 0.02], 
                     'ImagePositionPatient': [0.0, 0.1, 0.3], 
                     'ImageOrientationPatient': [0.0,0.0,1.0,0.0,1.0,0.0], 
                     'SliceThickness': 0.04,
                     'SpacingBetweenSlices': 0.04}
        tagUpdateDict = {'SeriesNumber': 99, 
                         'StudyDescription': ([0x0008,0x1030], 'LO', "TestDataA"), 
                         'SerisDescription': ([0x0008,0x103e], 'LO', "SeriesWink"), 
                         'StudyID': ([0x0020,0x0010], 'SH', '1099')}
        dcmTK.writeNumpyArrayToDicom(pixArray[:,:,:3], None, patMatrix, tmpDir)
        if not DEBUG:
            shutil.rmtree(tmpDir)

        pixArray = np.load(imnpy)
        tmpDir = os.path.join(TEST_OUTPUT, 'tmp9')
        cleanMakeDirs(tmpDir)
        patMatrix = {'PixelSpacing': [0.02, 0.02], 
                     'ImagePositionPatient': [0.0, 0.1, 0.3], 
                     'ImageOrientationPatient': [0.0,0.0,1.0,0.0,1.0,0.0], 
                     'SliceThickness': 0.04,
                     'SpacingBetweenSlices': 0.04}
        tagUpdateDict = {'SeriesNumber': 88, 
                         'StudyDescription': ([0x0008,0x1030], 'LO', "TestDataB"), 
                         'SerisDescription': ([0x0008,0x103e], 'LO', "SeriesLaugh"), 
                         'StudyID': ([0x0020,0x0010], 'SH', '1088')}
        dcmTK.writeNumpyArrayToDicom(pixArray[:,:,3:], None, patMatrix, tmpDir, tagUpdateDict=tagUpdateDict)
        if not DEBUG:
            shutil.rmtree(tmpDir)


if __name__ == '__main__':
    unittest.main()
