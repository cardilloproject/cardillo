from setuptools import setup, find_packages

name = "cardillo"
version = "1.0.0"
author = "Jonas Breuling, Giuseppe Capobianco, Lisa Eberhardt, Simon Eugster"
author_email = (
    "breuling@inm.uni-stuttgart.de, giuseppe.capobianco@fau.de, eberhardt@inm.uni-stuttgart.de, s.r.eugster@tue.nl",
)
url = ""
description = "Python package for flexible multibody dynamic problems."
long_description = ""
license = "LICENSE"

setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    install_requires=[
        "numpy>=1.26.4",
        "scipy>=1.13.0",
        "matplotlib>=3.4.3",
        "black>=24.4.0",
        "tqdm>=4.62.3",
        "pytest",
        "dill>=0.3.7",
        "cachetools>=5.3.3",
        "trimesh>=4.0.5",
        "vtk>=9.3.0",
        "scipy_dae @ git+https://github.com/JonasBreuling/scipy_dae.git",
        "urdf_parser_py",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
)
