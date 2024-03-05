# Cardillo

<p align="center">
<a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202-blue" alt="Licensing"/></a>
<!-- <a href="https://gitlab.com/JonasHarsch/cardillo3/-/tree/main"><img src="https://gitlab.com/JonasHarsch/cardillo3/badges/main/pipeline.svg" alt="gitlab pipeline status"/></a> -->
<a href="https://www.python.org/dev/peps/pep-0008/"><img alt="Code standard: PEP8" src="https://img.shields.io/badge/code%20standard-PEP8-black"></a>
</p>

## Table of Contents

- [Cardillo](#cardillo)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Virtual environments](#virtual-environments)
  - [Documentation](#documentation)
  - [Developing](#developing)
    - [Pre-requisites](#pre-requisites)
    - [Naming Conventions](#naming-conventions)
    - [Useful links](#useful-links)
    - [Further information](#further-information)

## Requirements
* Python 3.x (tested on Python 3.10 for Ubuntu 22.04.2 LTS)

## Installation
To install the package, clone or download the current repository, open a console, navigate to the root folder of the project and run `pip install .`.

```bash
git clone https://github.tik.uni-stuttgart.de/inm-cardillo/cardillo.git
cd cardillo3
pip install -e .
```

## Virtual environments
We recommend to work with virtual environments. These provide a Python environment that is isolated from the global Python installation. It can be set up using the following steps.

1. Create a virtual environment with the name `myvenv`
```bash
python -m venv myvenv
```
2. Activate the virtual environment by executing the `activate` script

macOS/Linux:
```bash
source ./myvenv/bin/activate
```

Windows:
```bash
myvenv/Scripts/activate
```
3. The Python path of your shell is now set to the Python installation in the virtual environment. Proceed to install cardillo as ususal.

```bash
pip install -e .
```
## Documentation
~~You can access the documentation via [the following link](https://jonasharsch.gitlab.io/cardillo3)~~.
 
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
git clone https://github.tik.uni-stuttgart.de/inm-cardillo/cardillo.git
cd cardillo3
pip install -e .
```

In doing so, the package is installed as a link to the current root folder. Consequently, changes in the root folder directly apply to the installed package and can be tested/run without installing the changes again.

### Naming Conventions

https://peps.python.org/pep-0008/#naming-conventions

* modules, function, variable: `this_is_great`
* classes: `MyClass`
* constants: `ABS_TOL`

**exceptions**:
* mechanics beats convention, e.g., `A_IB`instead of `a_ik` or `Exp_SO3` instead of `eps_so3`

### Import Conventions

1. external library imports
2. cardillo imports from outside module
3. module imports

Within these groups order imports alphabetically.

### Useful links
* [Information for MATLAB users](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html).

* A nice and simple [git introduction](https://rogerdudler.github.io/git-guide/index.html) and a short [git cheat sheet](https://about.gitlab.com/images/press/git-cheat-sheet.pdf).
* Conversation of LaTeX expressions to unicode: [https://www.unicodeit.net/](https://www.unicodeit.net/)

### Further information
go to [this folder](man)

### RGB codes for paraview
- TU/e red: R200, G33, B37
- FAU grey: R140, G159, B177
- green: R0, G255, B0
- blue: R0, G0, B127
