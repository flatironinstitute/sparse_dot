from setuptools import setup

DISTNAME = 'sparse_dot_mkl'
VERSION = '0.1'
DESCRIPTION = "Intel MKL wrapper for sparse matrix operations"
MAINTAINER = 'Chris Jackson'
MAINTAINER_EMAIL = 'cj59@nyu.edu'
URL = 'https://github.com/asistradition/sparse_dot'
DOWNLOAD_URL = ''
LICENSE = 'MIT'


setup(name=DISTNAME,
      version=VERSION,
      description=DESCRIPTION,
      url=URL,
      author=MAINTAINER,
      author_email=MAINTAINER_EMAIL,
      license=LICENSE,
      packages=['sparse_dot'],
      install_requires=['numpy', 'scipy'],
      zip_safe=True)
