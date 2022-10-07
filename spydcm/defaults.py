# -*- coding: utf-8 -*-

"""Some simple, customisable defaults
"""

DEFAULT_SERIES_OVERVIEW_TAG_LIST = ('SeriesNumber',
                                    'SeriesDescription',
                                    #'Pulse Sequence Name',
                                    0x0019109c,
                                    'SeriesInstanceUID')

DEFAULT_SUBJECT_OVERVIEW_TAG_LIST = ('PatientName',
                                    'PatientID',
                                    'PatientBirthDate',
                                    'PatientSex',
                                    'PatientAge',
                                    'PatientWeight')

DEFAULT_STUDY_OVERVIEW_TAG_LIST = ('StudyID',
                                    0x00431062,
                                    'StudyDescription',
                                    'StudyInstanceUID',
                                    'StudyDate',
                                    'MagneticFieldStrength',
                                    'ProtocolName')