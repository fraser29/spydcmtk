# -*- coding: utf-8 -*-

import configparser
import os
import sys

config = configparser.ConfigParser()

all_config_files = [os.path.join(os.curdir, '..','spydcmtk.conf'), 
                    os.path.join(os.path.expanduser("~"),'spydcmtk.conf'),
                    os.path.join(os.path.expanduser("~"),'.spydcmtk.conf'), 
                    os.path.join(os.path.expanduser("~"), '.config','spydcmtk.conf'),
                    os.path.join(os.environ.get("SPYDCMTK_CONF", ''),'spydcmtk.conf')]

config.read(all_config_files)

environment = config.get("app", "environment")
DEBUG = config.get("app", "debug")

path_items = config.items( "series_overview_tags" )
SERIES_OVERVIEW_TAG_LIST = []
for key, tagName in path_items:
    SERIES_OVERVIEW_TAG_LIST.append(tagName)
SERIES_OVERVIEW_TAG_LIST = list(set(SERIES_OVERVIEW_TAG_LIST))


path_items = config.items( "study_overview_tags" )
STUDY_OVERVIEW_TAG_LIST = []
for key, tagName in path_items:
    STUDY_OVERVIEW_TAG_LIST.append(tagName)
STUDY_OVERVIEW_TAG_LIST = list(set(STUDY_OVERVIEW_TAG_LIST))


path_items = config.items( "patient_overview_tags" )
SUBJECT_OVERVIEW_TAG_LIST = []
for key, tagName in path_items:
    SUBJECT_OVERVIEW_TAG_LIST.append(tagName)
SUBJECT_OVERVIEW_TAG_LIST = list(set(SUBJECT_OVERVIEW_TAG_LIST))