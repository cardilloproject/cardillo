from setuptools import setup, find_packages

name = "cardillo"
version = "3.0.0"
author = "Jonas Harsch, Giuseppe Capobianco, Simon Eugster"
author_email = (
    "harsch@inm.uni-stuttgart.de, giuseppe.capobianco@fau.de, eugster@inm.uni-stuttgart.de",
)
url = ""
description = "Python package for spatial nonlinear beam theories."
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
        "numpy>=1.21.3",
        "scipy>=1.6.1",
        "matplotlib>=3.4.3",
        "black>=22.1.0",
        "tqdm>=4.62.3",
        "meshio>=4.1.1",
        "meshzoo>=0.9.14",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
)
