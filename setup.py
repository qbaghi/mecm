import ez_setup
ez_setup.use_setuptools()

from setuptools import setup, find_packages

exec(open('mecm/version.py').read()) # loads __version__

setup(name='mecm',
      version=__version__,
      author='Quentin Baghi',
    description='',
    long_description=open('README.md').read(),
    license='see LICENSE.txt',
    keywords="",
    packages= find_packages(exclude='docs'),
    install_requires=['numpy','scipy','pyfftw'])
#    install_requires=['numpy','scipy','pyfftw','cvxopt','numba'])
