from setuptools import setup, find_packages

name = "cardillo"
version = "3.0.0"
author = "Jonas Harsch, Giuseppe Capobianco, Simon Eugster"
author_email = (
    "harsch@inm.uni-stuttgart.de, giuseppe.capobianco@fau.de, eugster@inm.uni-stuttgart.de",
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
        "meshio @ git+https://github.com/JonasHarsch/meshio.git@master#egg=meshio",  # may cause problems with "rich" -> quick fix it by installing "rich"
        # "meshio>=5.0.0", # beam export may not work with this!
        "pytest",
        "dill>=0.3.7",
        "cachetools",
        "trimesh>=4.0.5",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
)
