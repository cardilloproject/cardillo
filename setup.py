import os
from setuptools import setup, find_packages

# path to the packages __init__.py module in project source tree
init = os.path.join(os.path.dirname(__file__), 'cardillo', '__init__.py')

# extract __name__, __version__, __author__ from cardillo/__init__.py
# We could simply import it from the package but we cannot be sure that 
# this package is importable before installation is done.
# see https://thepythonguru.com/writing-packages-in-python/
name = eval(list(filter(lambda l: l.startswith('__name__'), open(init)))[0].split('=')[-1])
version = eval(list(filter(lambda l: l.startswith('__version__'), open(init)))[0].split('=')[-1])
author = eval(list(filter(lambda l: l.startswith('__author__'), open(init)))[0].split('=')[-1])
author_email = eval(list(filter(lambda l: l.startswith('__author_email__'), open(init)))[0].split('=')[-1])[0]

setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    description="cardillo's second version",
    long_description='',
    install_requires=[
        'numpy>=1.18.3', 
        'scipy>=1.4.1', 
        'matplotlib>=3.2.1', 
        'tqdm>=4.46.0',
        'meshio>=4.1.1', ],
    packages=find_packages(),
    python_requires='>=3.7',
)