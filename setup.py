#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'spydcmtk'
DESCRIPTION = 'Simple python dicom toolkit.'
URL = 'https://github.com/fraser29/spydcmtk'
EMAIL = 'callaghan.fm@gmail.com'
AUTHOR = 'Fraser M. Callaghan'
REQUIRES_PYTHON = '>=3.9.0'
VERSION = '1.2.3'
KEYWORDS="medical, imaging, mri, ct, dicom"

# What packages are required for this module to be executed?
REQUIRED = [
    'pydicom>=3.0.1', 'numpy>=1.26.2', 'tqdm>=4.66.1', 'vtk>=9.3.0', 'python-gdcm', 'highdicom', 'matplotlib', 'ngawari>=0.1.2'
]


here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION



# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    keywords=KEYWORDS,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    entry_points={
        "console_scripts": ["spydcmtk=spydcmtk.spydcm:main"],
    },
    package_data={NAME: ['spydcmtk.conf', 'ParaViewGlance.html']},
    install_requires=REQUIRED,
    # extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
