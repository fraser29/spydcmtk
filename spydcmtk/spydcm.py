# -*- coding: utf-8 -*-

"""Module that exposes the routines and utilities making up SPYDCMTK
"""

import os
import sys
import base64
import argparse
import numpy as np
import pydicom as dicom

# Local imports 
import spydcmtk.dcmTools as dcmTools
import spydcmtk.dcmTK as dcmTK
from spydcmtk.helpers import VTI_NAMING_TAG_LIST



def writeDirectoryToNII(dcmDir, outputPath, fileName):
    """ Write a directory of dicom files to a Nifti (*.nii.gz) file. 
        Uses dcm2nii so MUST have dcm2nii installed and within path.

    Args:
        dcmDir (str): Path to directory containing dicom files
        outputPath (str): Path to output directory where to save nifti
        fileName (str): Name of output nii.gz file (will rename nii.gz output from dcm2nii)
    """
    return dcmTools.writeDirectoryToNII(dcmDir, outputPath, fileName)


def buildTableOfDicomParamsForManuscript(topLevelDirectoryList, outputCSVPath, seriesDescriptionIdentifier=None, ONE_FILE_PER_DIR=True):
    """ Build a simple table of dicom parameters that would be suitable for input into a scientific manuscript.
        Output is comma delimited multiline string of Tr, Te (etc) stats. 

    Args:
        topLevelDirectoryList (list): list of str that are paths to directories containing dicom files
        outputCSVPath (str): path to outputCSV
        seriesDescriptionIdentifier (str, optional): A substring of SeriesDescription tag used to select certian series only. Defaults to None.
        ONE_FILE_PER_DIR (bool, optional): Set false to read all dicoms, can set True for faster reading if dicoms organised. Default: True
    """
    dfData = []
    for inspectDir in topLevelDirectoryList:
        try:
            ssL = dcmTK.ListOfDicomStudies.setFromDirectory(inspectDir, OVERVIEW=True, HIDE_PROGRESSBAR=True, ONE_FILE_PER_DIR=ONE_FILE_PER_DIR)
        except IndexError: # no dicoms found 
            continue
        for ss in ssL:
            if seriesDescriptionIdentifier is not None:
                matchingSeries = ss.getSeriesMatchingDescription([seriesDescriptionIdentifier], RETURN_SERIES_OBJ=True)
            else:
                matchingSeries = ss
            if matchingSeries is not None:
                for iSeries in matchingSeries:
                    resDict = iSeries.getSeriesInfoDict()
                    resDict["Identifier"] = dcmTools.getDicomFileIdentifierStr(iSeries[0])
                    dfData.append(resDict)
    stats = {}
    tagList = sorted(dfData[0].keys())
    strOut = ','+','.join(tagList) + '\n'
    for row in dfData:
        strOut += ','
        for i in tagList:
            strOut += f'{row[i]},'
            stats.setdefault(i, []).append(row[i])
        strOut += '\n'
    strOut += '\n\nSTATS:\n'
    for label, myFunc in zip(['Mean', 'Standard Deviation', 'Median', 'Min', 'Max'], 
                             [np.mean, np.std, np.median, np.min, np.max]):
        for k1 in range(len(tagList)):
            if k1 == 0:
                strOut += f'{label},'
            try:
                strOut += f'{myFunc(stats[tagList[k1]])},'
            except:
                strOut += 'NAN,'
        strOut += '\n'
    with open(outputCSVPath, 'w') as fid:
        fid.write(strOut)
    return outputCSVPath


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


def returnFirstDicomFound(rootDir, FILE_NAME_ONLY=False, MatchingTag_dict=None):
    """
    Search recursively for first dicom file under root and return.
    If have dicoms in nice folder structure then this can be a fast way to find, e.g. all series with protocol X

    :param rootDir: directory on filesystem
    :param FILE_NAME_ONLY: If true will return the file name [Default False]
    :param MatchingTag_dict: If given then will only consider dicoms where tag(key) matches value given [Default None]
    :return: pydicom dataset<without pixel data> or fileName<str>
    """
    for root, _, files in os.walk(rootDir):
        for iFile in files:
            if 'dicomdir' in iFile.lower():
                continue
            thisFile = os.path.join(root, iFile)
            try:
                dataset = dicom.read_file(thisFile, stop_before_pixels=True)
                if MatchingTag_dict is not None:
                    tf = []
                    for iKey in MatchingTag_dict.keys():
                        tf.append(str(dataset[iKey].value) == str(MatchingTag_dict[iKey]))
                    if not all(tf):
                        continue
                if FILE_NAME_ONLY:
                    return thisFile
                else:
                    return dataset
            except dicom.filereader.InvalidDicomError:
                continue
    return None

def directoryToVTI(dcmDirectory, outputFolder, 
                   outputNamingTags=VTI_NAMING_TAG_LIST, 
                   QUITE=True, FORCE=False, INCLUDE_MATRIX=True):
    """Convert directory of dicoms to VTI files (one vti per series)
        Naming built from dicom tags: 

    Args:
        dcmDirectory (str): Directory containing dicom files
        outputFolder (str): Directory where vti output files to be written
        outputNamingTags (tuple, optional): Dicom tags used to generate vti file name. 
                        Defaults to ('PatientName', 'SeriesNumber', 'SeriesDescription')
        QUITE (bool, optional): Suppress output information. Defaults to True.
        FORCE (bool, optional): Set True to overwrite already present files. Defaults to False.

    Returns:
        list: List of output file names written
    """
    ListDicomStudies = dcmTK.ListOfDicomStudies.setFromInput(dcmDirectory, HIDE_PROGRESSBAR=QUITE, FORCE_READ=FORCE, OVERVIEW=False) 
    return __listDicomStudiesToVTI(ListDicomStudies, outputFolder=outputFolder, outputNamingTags=outputNamingTags, INCLUDE_MATRIX=INCLUDE_MATRIX)


def directoryToVTS(dcmDirectory, outputFolder,
                   outputNamingTags=VTI_NAMING_TAG_LIST, 
                   QUITE=True, FORCE=False):
    """Convert directory of dicoms to VTS files (Good for cine etc)
        Naming built from dicom tags: 

    Args:
        dcmDirectory (str): Directory containing dicom files
        outputFolder (str): Directory where vti output files to be written
        outputNamingTags (tuple, optional): Dicom tags used to generate vti file name. 
                        Defaults to ('PatientName', 'SeriesNumber', 'SeriesDescription')
        QUITE (bool, optional): Suppress output information. Defaults to True.
        FORCE (bool, optional): Set True to overwrite already present files. Defaults to False.

    Returns:
        list: List of output file names written
    """
    ListDicomStudies = dcmTK.ListOfDicomStudies.setFromInput(dcmDirectory, HIDE_PROGRESSBAR=QUITE, FORCE_READ=FORCE, OVERVIEW=False) 
    return __listDicomStudiesToVTI(ListDicomStudies, outputFolder=outputFolder, outputNamingTags=outputNamingTags, VTS=True)


def __listDicomStudiesToVTI(ListDicomStudies, outputFolder, outputNamingTags=VTI_NAMING_TAG_LIST, QUIET=True, INCLUDE_MATRIX=True, VTS=False):
    outputFiles = []
    for iDS in ListDicomStudies:
        for iSeries in iDS:
            if VTS:
                print(f"DICOMS TO VTS IS NOT IMPLEMENTED YET")
                # fOut = iSeries.writeToVTS(outputPath=outputFolder, outputNamingTags=outputNamingTags)
            else:
                fOut = iSeries.writeToVTI(outputPath=outputFolder, outputNamingTags=outputNamingTags, INCLUDE_MATRIX=INCLUDE_MATRIX)
            outputFiles.append(fOut)
            if not QUIET:
                print(f'Written {fOut}')
    return outputFiles

# =========================================================================
# =========================================================================
## DATA TO HTML
# =========================================================================
def convertInputsToHTML(listOfFilePaths, outputFile=None, glanceHtml=None, QUIET=False, DEBUG=False):
    """Convert inputs to HTML viewable via ParaViewGlance (3D volumes and/or surface polymeshes)
    
    Keyword arguments:
    listOfFilePaths             -- May be nii, vtk or dicoms
    outputFile (optional)       -- Full path to outputfile, if not given then use first input path. Defaults to None.
    glanceHtml (optional)       -- Full path to glanceHtml template file, if not given then use default. Defaults to None.
    QUIET (optional)            -- Set True to suppress output. Defaults to False.
    DEBUG (optional)            -- Set True to prevent cleanup of intermediary files. Defaults to False.
    Return: path to html file written
    """
    
    CLEAN_UP_LIST = []
    FILE_TO_VTK_LIST = []
    ## --- Check inputs ---
    # Glance html
    if glanceHtml is None:
        thisDir = os.path.split(os.path.realpath(__file__))[0]
        glanceHtml = os.path.join(thisDir, 'ParaViewGlance.html')
        if not QUIET:
            print('Using ParaView glance file: %s'%(glanceHtml))
    if not os.path.isfile(glanceHtml):
        raise ValueError('%s does not exist'%(glanceHtml))
    if type(listOfFilePaths) != list:
        listOfFilePaths = [listOfFilePaths]
    ## --- Output file ---
    if outputFile is None:
        outputDir, fName = os.path.split(listOfFilePaths[0])
        fNameOut = os.path.splitext(fName)[0]+'.html'
        outputFile = os.path.join(outputDir, fNameOut)
    elif os.path.isdir(outputFile):
        outputDir = outputFile
        _, fName = os.path.split(listOfFilePaths[0])
        fNameOut = os.path.splitext(fName)[0]+'.html'
        outputFile = os.path.join(outputDir, fNameOut)
    else:
        outputDir, fNameOut = os.path.split(outputFile)

    ## --- VTK Objs / file paths ---
    for iPath in listOfFilePaths:
        if os.path.isfile(iPath):
            if iPath.endswith('nii'):
                iPath = dcmTK.dcmVTKTK.nii2vti(iPath)
                CLEAN_UP_LIST.append(iPath)
            FILE_TO_VTK_LIST.append(iPath)
        else:
            if os.path.isdir(iPath): # If path to dicoms
                dcmToVTKPath = directoryToVTI(iPath, outputDir, INCLUDE_MATRIX=False)
                for ifile in dcmToVTKPath:
                    if ifile.endswith('.pvd'):
                        FILE_TO_VTK_LIST += list(dcmTK.dcmVTKTK.readPVDFileName(ifile).values())
                    else:
                        FILE_TO_VTK_LIST.append(ifile)
                CLEAN_UP_LIST += dcmToVTKPath
            else:
                raise ValueError('%s does not exist' % (iPath))

    ## --- Build HTML Recursivly ---
    if not QUIET:
        print('Writing %s from base html %s, using:'%(outputFile, glanceHtml))
        for iFile in listOfFilePaths:
            print('    %s'%(iFile))

    for k1, iFile in enumerate(FILE_TO_VTK_LIST):
        outputTemp = outputFile[:-5]+'_TEMP%d.html'%(k1)
        glanceHtml = __vtkToHTML(iFile, glanceHtml, outputTemp)
        CLEAN_UP_LIST.append(outputTemp)
    os.rename(CLEAN_UP_LIST.pop(), outputFile)

    ## --- Clean up --- // Skipped if in DEBUG mode
    if not DEBUG:
        if not QUIET:
            print('Cleaning up:', str(CLEAN_UP_LIST))
        for ifile in CLEAN_UP_LIST:
            if iFile.endswith('.pvd'):
                dcmTK.dcmVTKTK.deleteFilesByPVD(iFile)
            else:
                os.unlink(ifile)

    return outputFile


def __vtkToHTML(vtkDataPath, glanceHtml, outputFile):
    # Extract data as base64
    with open(vtkDataPath, "rb") as data:
        dataContent = data.read()
        base64Content = base64.b64encode(dataContent)
        base64Content = base64Content.decode().replace("\n", "")
    # Create new output file
    with open(glanceHtml, mode="r", encoding="utf-8") as srcHtml:
        with open(outputFile, mode="w", encoding="utf-8") as dstHtml:
            for line in srcHtml:
                if "</body>" in line:
                    dstHtml.write("<script>\n")
                    dstHtml.write('var contentToLoad = "%s";\n\n' % base64Content)
                    dstHtml.write(
                        'Glance.importBase64Dataset("%s" , contentToLoad, glanceInstance.proxyManager);\n'
                        % os.path.basename(vtkDataPath)
                    )
                    dstHtml.write("glanceInstance.showApp();\n")
                    dstHtml.write("</script>\n")
                dstHtml.write(line)
    return outputFile


### ====================================================================================================================
##          RUN VIA MAIN
### ====================================================================================================================
# Override error to show help on argparse error (missing required argument etc)
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
##  ========= INSPECTION / VERIFICATION =========
def quickInspect(dStudies, FULL):
    frmtStr = dStudies.getSummaryString(FULL)
    print(frmtStr)

def checkArgs(args):
    allActionArgs = [args.nii, 
                # args.vti, 
                args.quickInspect, 
                args.quickInspectFull, 
                args.outputFolder is not None]
    return any(allActionArgs)

##  ========= RUN ACTIONS =========
def runActions(args, ap):

    ####
    if args.dcmdump:
        ds = returnFirstDicomFound(args.inputPath, FILE_NAME_ONLY=False)
        print(ds)
    elif args.msTable:
        fOut = buildTableOfDicomParamsForManuscript([args.inputPath], 
                                                    outputCSVPath=os.path.join(args.outputFolder, 'ms.csv'), 
                                                    seriesDescriptionIdentifier=None,
                                                    ONE_FILE_PER_DIR=False)
        print(f"Written Manuscript table like csv to {fOut}")
    else:
        # check arguments to avoid reading all dicoms and then doing nothing...
        if not checkArgs(args):
            ap.exit(0, f'No action given. Exiting SPYDCMTK without action\n')
        if args.STREAM:
            dcmTools.streamDicoms(args.inputPath, args.outputFolder, FORCE_READ=args.FORCE, HIDE_PROGRESSBAR=args.QUIET, SAFE_NAMING=args.SAFE)
            return 0
        try:
            onlyOverview = args.quickInspect or args.quickInspectFull
            if not args.QUIET:
                print(f"READING...")
            ListDicomStudies = dcmTK.ListOfDicomStudies.setFromInput(args.inputPath, HIDE_PROGRESSBAR=args.QUIET, FORCE_READ=args.FORCE, OVERVIEW=onlyOverview) 
            if args.SAFE:
                ListDicomStudies.setSafeNameMode()
        except IOError as e:
            ap.exit(1, f'Error reading {args.inputPath}.\n    {e}')
            # Let IOERROR play out here is not correct input
        ##
        if args.quickInspect or args.quickInspectFull:
            for iStudy in ListDicomStudies:
                print(iStudy[0].getRootDir())
                print(iStudy.getStudySummary(args.quickInspectFull))
                print('\n')
        else:
            if args.outputFolder is None:
                print(f'WARNING: outputFolder not given - setting to inputFolder')
                args.outputFolder = os.path.split(args.inputPath)[0]
            if args.nii:
                for iDS in ListDicomStudies:
                    for iSeries in iDS:
                        fOut = iSeries.writeToNII(outputPath=args.outputFolder)
                        if not args.QUIET:
                            print(f'Written {fOut}')
            # elif args.vti:
            #     __listDicomStudiesToVTI(ListDicomStudies=ListDicomStudies, outputFolder=args.outputFolder, QUIET=args.QUIET, INCLUDE_MATRIX=(not args.NO_MATRIX))
            elif args.html:
                for iDS in ListDicomStudies:
                    for iSeries in iDS:
                        fOut = convertInputsToHTML([iSeries.getRootDir()], args.outputFolder)
                        if not args.QUIET:
                            print(f'Written {fOut}')
            elif args.outputFolder is not None:
                if not args.QUIET:
                    print(f"WRITTING...")
                outDirList = ListDicomStudies.writeToOrganisedFileStructure(args.outputFolder, anonName=args.anonName, anonID=args.anonID)
                allDirsPresent = all([os.path.isdir(i) for i in outDirList])
                res = 0 if allDirsPresent else 1
                ap.exit(res, f'Transfer and sort from {args.inputPath} to {args.outputFolder} COMPLETE\n')
        ##

### ====================================================================================================================
### ====================================================================================================================
# S T A R T
#
def main():
    # --------------------------------------------------------------------------
    #  ARGUMENT PARSING
    # --------------------------------------------------------------------------
    ap = MyParser(description='Simple Python Dicom Toolkit - spydcmtk')

    ap.add_argument('-i', dest='inputPath', help='Path to find dicoms (file or directory or tar or tar.gz or zip)', type=str, required=True)
    ap.add_argument('-o', dest='outputFolder', help='Path for output - if set then will organise dicoms into this folder', type=str, default=None)

    ap.add_argument('-a', dest='anonName',
        help='anonymous name [optional - if not given, then not anoymised]', type=str, default=None)
    ap.add_argument('-aid', dest='anonID',
        help='anonymous ID [optional - only used if anonName is given, default=""]', type=str, default='')
    ap.add_argument('-quickInspect', dest='quickInspect',
        help='Will output a summary of dicoms to the terminal', action='store_true')
    ap.add_argument('-quickInspectFull', dest='quickInspectFull',
        help='Will output a full summary of dicoms to the terminal', action='store_true')
    ap.add_argument('-msTable', dest='msTable',
        help='Will output a csv to outputFolder with tags info suitable for building manuscript style table', action='store_true')
    ap.add_argument('-dcmdump', dest='dcmdump',
        help='Will output a dump of all dicom tags to the terminal (from first found dicom)', action='store_true')
    ap.add_argument('-nii', dest='nii',
        help='Will convert each series to nii.gz. Naming: {PName}_{SE#}_{SEDesc}.nii.gz', action='store_true')
    # ap.add_argument('-vti', dest='vti',
    #     help='Will convert each series to vti. Naming: {PName}_{SE#}_{SEDesc}.vti', action='store_true')
    # ap.add_argument('-NO_MATRIX', dest='NO_MATRIX', help='No matrix added to vti files', action='store_true')
    ap.add_argument('-html', dest='html',
        help='Will convert each series to html file for web viewing. Naming: outputfolder argument', action='store_true')
    # -- program behaviour guidence -- #
    ap.add_argument('-SAFE', dest='SAFE', help='Safe naming - uses UIDs for naming to avoid potential conflicts.\n\t'+\
                        'Note, normal behaviour is to check for safe naming but certain conditions may require this.',
                            action='store_true')
    ap.add_argument('-FORCE', dest='FORCE', help='force reading even if not standard dicom (needed if dicom files missing header meta)',
                            action='store_true')
    ap.add_argument('-QUIET', dest='QUIET', help='Suppress progress bars and information output to terminal',
                            action='store_true')
    ap.add_argument('-STREAM', dest='STREAM', help='To organise dicoms rapidly without any quality checking',
                            action='store_true')
    ##

    arguments = ap.parse_args()
    if arguments.inputPath is not None:
        arguments.inputPath = os.path.abspath(arguments.inputPath)
        if not (os.path.isdir(arguments.inputPath) or os.path.isfile(arguments.inputPath)):
            print(f'## ERROR : Can not find input {arguments.inputPath}')
            print('EXITING')
            sys.exit(1)
        if not arguments.QUIET:
            print(f'Running SPYDCMTK with input {arguments.inputPath}')
    ## -------------

    runActions(arguments, ap)


if __name__ == '__main__':

    main()