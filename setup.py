from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import codecs
import os
import sys
import re

import svmlitepy

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(os.path.join(here, *parts), 'r').read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        # self.test_args = ['--strict', '--verbose', '--tb=long', 'tests']
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

long_description = read('README.rst')

setup(
    name='svmlitepy',
    version=find_version('svmlitepy', '__init__.py'),
    url='https://github.com/DushyantaDhyani/svmlitepy/',
    license='MIT License',
    author='Dushyanta Dhyani',
    tests_require=['pytest'],
    install_requires=['numpy>=1.10.4'
                      ],
    cmdclass={'test': PyTest},
    author_email='dush.dhyani@gmail.com',
    description='Machine Learning Library for Support Vector Classification using Sequential Minimal Optimization',
    long_description=long_description,
    # entry_points={
    #     'console_scripts': [
    #         'sandmanctl = sandman.sandmanctl:run',
    #         ],
    #     },
    packages=['svmlitepy'],
    include_package_data=True,
    platforms='any',
    # test_suite='sandman.test.test_sandman',
    zip_safe=False,
    # package_data={'sandman': ['templates/**', 'static/*/*']},
    classifiers=['Intended Audience :: Science/Research',
                                 "Intended Audience :: Developers",
                                 "Intended Audience :: Education",
                                 "Intended Audience :: Science/Research",
                                 'Programming Language :: Python',
                                 "Topic :: Scientific/Engineering :: Artificial Intelligence",
                                 'Topic :: Software Development',
                                 'Topic :: Scientific/Engineering',
                                 'Operating System :: Microsoft :: Windows',
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Operating System :: MacOS',
                                 'Programming Language :: Python :: 2.7',
                                 'License :: OSI Approved :: MIT License'
                                 ],
    extras_require={
        'testing': ['pytest'],
      }
)
