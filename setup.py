#!/usr/bin/env python3
import setuptools
import sys


import spectracl

requirements = open("requirements.txt").read().split()

setuptools.setup(
    name='spectracl',
    version=spectracl.__version__,
    description='Model-based classification of maldi-tof spectra',
    author='Stephen Watts',
    license='GPLv3',
    test_suite='tests',
    packages=setuptools.find_packages(),
    install_requires=requirements,
    package_data={'spectracl': ['data/features_selected.txt', 'data/model.bin']},
    entry_points={
        'console_scripts': ['spectracl=spectracl.__main__:entry'],
    }
)
