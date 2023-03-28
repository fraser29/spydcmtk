
from context import spydcm

import os
import unittest

from spydcm import dcmTK


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
        


class TestDicomStudy(unittest.TestCase):
    def runTest(self):
        dcmStudy = dcmTK.DicomStudy.setFromDirectory(TEST_DIRECTORY, HIDE_PROGRESSBAR=True)
        self.assertEqual(len(dcmStudy), 1, "Incorrect number series in dcmStudy")


class TestDicom2VT2Dicom(unittest.TestCase):
    def runTest(self):
        dcmStudy = dcmTK.DicomStudy.setFromDirectory(TEST_DIRECTORY, HIDE_PROGRESSBAR=True)
        dcmSeries = dcmStudy.getSeriesBySeriesNumber(41)
        vtiDict = dcmSeries.buildVTIDict()
        self.assertAlmostEqual(list(vtiDict.keys())[1], 0.05192, places=7, msg='time key in vti dict incorrect')
        print(dcmSeries.writeToVTI('/home/fraser/temp'))



if __name__ == '__main__':
    unittest.main()
