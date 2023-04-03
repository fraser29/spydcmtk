# -*- coding: utf-8 -*-

import configparser
from collections import OrderedDict
import os

thisConfFileName = 'spydcmtk.conf'
rootDir = os.path.abspath(os.path.dirname(__file__))

def group_items_to_list(config, default_items, groupName):
    path_items = config.items( groupName )
    myList = []
    for __, tagName in path_items:
        if tagName in default_items:
            continue
        if tagName not in myList: # Avoid doubles
            myList.append(tagName)
    return myList

config = configparser.ConfigParser(dict_type=OrderedDict)

all_config_files = [os.path.join(rootDir,thisConfFileName), 
                    os.path.join(os.path.expanduser("~"),thisConfFileName),
                    os.path.join(os.path.expanduser("~"),'.'+thisConfFileName), 
                    os.path.join(os.path.expanduser("~"), '.config',thisConfFileName),
                    os.environ.get("SPYDCMTK_CONF", '')]

config.read(all_config_files)

environment = config.get("app", "environment")
DEBUG = config.get("app", "debug")

default_items = [vv for _,vv in config.items('DEFAULT')]


SERIES_OVERVIEW_TAG_LIST = group_items_to_list(config, default_items, "series_overview_tags")
STUDY_OVERVIEW_TAG_LIST = group_items_to_list(config, default_items, "study_overview_tags")
SUBJECT_OVERVIEW_TAG_LIST = group_items_to_list(config, default_items, "patient_overview_tags")
