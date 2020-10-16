#!/usr/bin/env python3
import setuptools
import sys


import spectracl


setuptools.setup(
    name='spectracl',
    version=spectracl.__version__,
    description='Model-based classification of maldi-tof spectra',
    author='Stephen Watts',
    license='GPLv3',
    test_suite='tests',
    packages=setuptools.find_packages(),
    package_data={'spectracl': ['data/species_map.tsv', 'data/model.bin']},
    entry_points={
        'console_scripts': ['spectracl=spectracl.__main__:entry'],
    }
)
