"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding

# Automatically get package dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="rtn",  # Required
    version="0.1.0",  # Required
    # package_dir={'': 'src'},  # Optional
    packages=find_packages(),  # Required
    python_requires=">=3.7",
    # packages=['src'],
    install_requires=requirements,
)
