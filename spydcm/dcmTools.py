# -*- coding: utf-8 -*-

"""Basic helper tools here
"""

import os
import datetime
import tarfile
import glob
import json
import datetime
import numpy as np


def __countFilesInDir(dirName):
    files = []
    if os.path.isdir(dirName):
        for path, dirs, filenames in os.walk(dirName):  # @UnusedVariable
            files.extend(filenames)
    return len(files)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        self.ensure_ascii = False
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def writeDictionaryToJSON(fileName, dictToWrite):
    with open(fileName, 'w') as fp:
        json.dump(dictToWrite, fp, indent=4, sort_keys=True, cls=NumpyEncoder, ensure_ascii=False)
    return fileName

def parseJsonToDictionary(fileName):
    with open(fileName, 'r') as fid:
        myDict = json.load(fid)
    return myDict

def fixPath(p):
    return p.encode('utf8', 'ignore').strip().decode()

def cleanString(ss):
    if not type(ss) == str:
        return ss
    ss = ss.replace('^', '-')
    ss = ss.replace(' ', '_')
    keepcharacters = ('-', '.', '_', 'ö','ü','ä','é','è','à')
    ss = "".join([c for c in ss if (c.isalnum() or (c.lower() in keepcharacters))]).rstrip()
    try:
        if ss[-1] == '.':
            ss = ss[:-1]
    except IndexError:
        pass
    # return ss
    return fixPath(ss)


def dbDateToDateTime(dbDate):
    try:
        return datetime.datetime.strptime(dbDate, '%Y%m%d')
    except ValueError:
        return datetime.datetime.strptime(dbDate, '%Y%m%dT%H%M%S')


def distPts(pt1, pt2):
    try:
        dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2)
    except IndexError:
        dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    return dist


def _isCompressed(ds):
    """
    Check if dicom dataset is compressed or not
    """
    uncompressed_types = ["1.2.840.10008.1.2",
                            "1.2.840.10008.1.2.1",
                            "1.2.840.10008.1.2.1.99",
                            "1.2.840.10008.1.2.2"]

    if ('TransferSyntaxUID' in ds.file_meta) and (ds.file_meta.TransferSyntaxUID in uncompressed_types):
        return False
    elif 'TransferSyntaxUID' not in ds.file_meta:
        return False
    return True

def instanceNumberSortKey(val):
    try:
        return int(__getTags(val, ['InstanceNumber'])['InstanceNumber'])
    except (ValueError, IOError, AttributeError):
        return 99e99

def __getTags(dataset, tagsList):
    tagsDict = {}
    for iKey in tagsList:
        tagsDict[iKey] = dataset.get(iKey, 'Unknown')
    return tagsDict


def walkdir(folder):
    """Walk through each files in a directory"""
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            yield os.path.abspath(os.path.join(dirpath, filename))
