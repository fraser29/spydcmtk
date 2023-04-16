
from context import spydcmtk # This is useful for testing outside of environment

import os
import unittest
import shutil
from spydcmtk import dcmTK
from spydcmtk import spydcm


this_dir = os.path.split(os.path.realpath(__file__))[0]
TEST_DIRECTORY = os.path.join(this_dir, 'TEST_DATA')
dcm001 = os.path.join(TEST_DIRECTORY, 'IM-00041-00001.dcm')


class TestDicomSeries(unittest.TestCase):
    def runTest(self):
        dcmSeries = dcmTK.DicomSeries.setFromDirectory(TEST_DIRECTORY, HIDE_PROGRESSBAR=True)
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
        tmpDir = os.path.join(TEST_DIRECTORY, 'tmp1')
        os.makedirs(tmpDir)
        dcmSeries.writeToOrganisedFileStructure(tmpDir)
        dcmSeries.writeToOrganisedFileStructure(tmpDir, anonName='Not A Name')
        shutil.rmtree(tmpDir)
        


class TestDicomStudy(unittest.TestCase):
    def runTest(self):
        dcmStudy = dcmTK.DicomStudy.setFromDirectory(TEST_DIRECTORY, HIDE_PROGRESSBAR=True)
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
        tmpDir = os.path.join(TEST_DIRECTORY, 'tmp2')
        os.makedirs(tmpDir)
        dcmStudy.writeToOrganisedFileStructure(tmpDir)
        dcmStudy.writeToOrganisedFileStructure(tmpDir, anonName='Not A Name')
        shutil.rmtree(tmpDir)


class TestDicom2VT2Dicom(unittest.TestCase):
    def runTest(self):
        dcmStudy = dcmTK.DicomStudy.setFromDirectory(TEST_DIRECTORY, HIDE_PROGRESSBAR=True)
        dcmSeries = dcmStudy.getSeriesBySeriesNumber(41)
        vtiDict = dcmSeries.buildVTIDict()
        self.assertAlmostEqual(list(vtiDict.keys())[1], 0.05192, places=7, msg='time key in vti dict incorrect')
        fOut = dcmSeries.writeToVTI('/tmp')
        self.assertTrue(os.path.isfile(fOut), msg='Written pvd file does not exist')
        dd = dcmTK.dcmVTKTK.readPVD(fOut)
        dTimes = sorted(dd.keys())
        self.assertAlmostEqual(dTimes[1], 0.05192, places=7, msg='time key in vti dict incorrect')
        vti0 = dd[dTimes[0]]
        oo = vti0.GetOrigin()
        self.assertAlmostEqual(oo[1], 0.1166883275304, places=7, msg='origin in vti dict incorrect')
        dcmTK.dcmVTKTK.deleteFilesByPVD(fOut)


class TestDicom2MSTable(unittest.TestCase):
    def runTest(self):
        msTable = spydcm.buildTableOfDicomParamsForManuscript([this_dir], 'a')
        topDir = '/Volume/MRI_DATA'
        if os.path.isdir(topDir):
            msTable = spydcm.buildTableOfDicomParamsForManuscript([os.path.join(topDir, i) for i in os.listdir(topDir)], '4CH')
        


if __name__ == '__main__':
    unittest.main()
