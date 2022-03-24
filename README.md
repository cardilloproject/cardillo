# Cardillo

<p align="center">
<a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202-blue" alt="Licensing"/></a>
<a href="https://gitlab.com/JonasHarsch/cardillo3/-/tree/main"><img src="https://gitlab.com/JonasHarsch/cardillo3/badges/main/pipeline.svg" alt="gitlab pipeline status"/></a>
<a href="https://www.python.org/dev/peps/pep-0008/"><img alt="Code standard: PEP8" src="https://img.shields.io/badge/code%20standard-PEP8-black"></a>
</p>

## Table of Contents

- [PyNSD](#pynsd)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Documentation](#documentation)
  - [Folder hierarchy](#folder-hierarchy)
  - [Running a scenario](#running-a-scenario)
  - [Developing](#developing)
    - [Pre-requisites](#pre-requisites)
    - [Useful links](#useful-links)
    - [Developing a new system](#developing-a-new-system)
    - [Developing a new solver](#developing-a-new-solver)

## Requirements
* Python 3.x (tested on Python 3.8 for Ubuntu 20.04.4 LTS)

## Installation
To install the package, clone or download the current repository, open a console, navigate to the root folder of the project and run `pip install .`.

```bash
git clone https://gitlab.com/JonasHarsch/cardillo3.git
cd cardillo3
pip install .
```

## Documentation
You can access the documentation via [the following link](https://jonasharsch.gitlab.io/cardillo3.
 
## Developing

### Pre-requisites

In order to collaborate with the repository, you need to install `pre-commit` git hook, to correctly format and check the Python before pushing changes to a remote branch of the repository. To do so, execute the following code in a console in the project root folder:

```bash
pip install pre-commit
pre-commit install
```

Now, `pre-commit` will run on every commit applying format to the files, and thus passing the CI/CD pipeline without any errors.

Moreover, it is advised to install the package in editable mode using `pip install -e .`, i.e.,

```bash
git clone https://gitlab.com/JonasHarsch/cardillo3.git
cd cardillo3
pip install -e .
```

In doing so, the package is installed as a link to the current root folder. Consequently, changes in the root folder directly apply to the installed package and can be tested/run without installing the changes again.

### Useful links
[information for MATLAB users](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html).

The [PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/) should be complied.

A nice and simple [git introduction](https://rogerdudler.github.io/git-guide/index.html) and a short [git cheat sheet](https://about.gitlab.com/images/press/git-cheat-sheet.pdf).
