# -*- coding: utf-8 -*-

import configparser
import json
from collections import OrderedDict
import os


thisConfFileName = 'spydcmtk.conf'
rootDir = os.path.abspath(os.path.dirname(__file__))


class _SpydcmTK_config():
    def __init__(self, ) -> None:
        

        self.config = configparser.ConfigParser(dict_type=OrderedDict)
        self.all_config_files = [os.path.join(rootDir,thisConfFileName), 
                            os.path.join(os.path.expanduser("~"),thisConfFileName),
                            os.path.join(os.path.expanduser("~"),'.'+thisConfFileName), 
                            os.path.join(os.path.expanduser("~"), '.config',thisConfFileName),
                            os.environ.get("SPYDCMTK_CONF", '')]

    def runconfigParser(self, extraConfFile=None):
        
        if extraConfFile is not None:
            if os.path.isfile(extraConfFile):
                self.all_config_files.append(extraConfFile)
            else:
                print(f"WARNING: {extraConfFile} passed as config file to read, but FileNotFound")

        self.config.read(self.all_config_files)

        self.environment = self.config.get("app", "environment")
        self.DEBUG = self.config.getboolean("app", "debug", fallback=False)
        self.dcm2nii_path = self.config.get("app", "dcm2nii_path")
        self.dcm2nii_options = self.config.get("app", "dcm2nii_options", fallback='')

        # default_items = [vv for _,vv in self.config.items('DEFAULT')]
        extra_error_info = "Ensure to use double not single quotes for strings. "

        ## READ TAGS LISTS FOR INFO / OUTPUT NAMING
        ConfigGroupList = ["SERIES_OVERVIEW_TAG_LIST", 
                           "STUDY_OVERVIEW_TAG_LIST",
                           "SUBJECT_OVERVIEW_TAG_LIST",
                           "MANUSCRIPT_TABLE_EXTRA_TAG_LIST",
                           "VTI_NAMING_TAG_LIST",
                           "SUBJECT_NAMING_TAG_LIST",
                           "STUDY_NAMING_TAG_LIST",
                           "SERIES_NAMING_TAG_LIST"]
        for iGroup in ConfigGroupList:

            try:
                setattr(self, iGroup, json.loads(self.config.get(iGroup.lower(), "tagList")))
            except json.decoder.JSONDecodeError:
                print(f"Error reading {iGroup} from spydcmtk.conf.")
                print(f"  {extra_error_info}")



        # try:
        #     self.SERIES_OVERVIEW_TAG_LIST  = json.loads(self.config.get("series_overview_tags", "tagList"))
        # except json.decoder.JSONDecodeError:
        #     print(f"Error reading series_overview_tags from spydcmtk.conf.")
        #     print(f"  {extra_error_info}")
        # try:
        #     self.STUDY_OVERVIEW_TAG_LIST   = json.loads(self.config.get("study_overview_tags", "tagList"))
        # except json.decoder.JSONDecodeError:
        #     print(f"Error reading study_overview_tags from spydcmtk.conf.")
        #     print(f"  {extra_error_info}")
        # try:
        #     self.SUBJECT_OVERVIEW_TAG_LIST = json.loads(self.config.get("patient_overview_tags", "tagList"))
        # except json.decoder.JSONDecodeError:
        #     print(f"Error reading patient_overview_tags from spydcmtk.conf.")
        #     print(f"  {extra_error_info}")
        # try: 
        #     self.VTI_NAMING_TAG_LIST       = json.loads(self.config.get("vti_naming_tags", "tagList"))
        # except json.decoder.JSONDecodeError:
        #     print(f"Error reading vti_naming_tags from spydcmtk.conf.")
        #     print(f"  {extra_error_info}")
        # try: 
        #     self.MANUSCRIPT_TABLE_EXTRA_TAG_LIST = json.loads(self.config.get("manuscript_table_extra_tags","tagList"))
        # except json.decoder.JSONDecodeError:
        #     print(f"Error reading manuscript_table_extra_tags from spydcmtk.conf.")
        #     print(f"  {extra_error_info}")
        # try: 
        #     self.SUBJECT_NAMING_TAG_LIST = json.loads(self.config.get("manuscript_table_extra_tags","tagList"))
        # except json.decoder.JSONDecodeError:
        #     print(f"Error reading manuscript_table_extra_tags from spydcmtk.conf.")
        #     print(f"  {extra_error_info}")

SpydcmTK_config = _SpydcmTK_config()
SpydcmTK_config.runconfigParser()