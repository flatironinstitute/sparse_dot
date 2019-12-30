from setuptools import setup, find_packages

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
      packages=find_packages(include=['sparse_dot_mkl', "sparse_dot_mkl.*"], exclude=["tests", "*.tests"]),
      install_requires=['numpy', 'scipy'],
      tests_require=['nose', 'coverage'],
      test_suite="nose.collector",
      zip_safe=True)
