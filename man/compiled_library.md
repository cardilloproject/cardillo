## Build a compiled library: ##

The following steps are taken from [this article](https://medium.com/analytics-vidhya/how-to-create-a-python-library-7d5aea80cc3f).

- Create a virtual environment following [these instructions](../README.md/#virtual-environments)
- install prerequisites:
  ```bash
    pip install wheel
    pip install setuptools
    pip install twine
  ``` 
- download cardillo and build library
  ```bash
    git clone git@github.tik.uni-stuttgart.de:inm-cardillo/cardillo.git
    python setup.py bdist_wheel
  ```
- install library
  ```bash
    pip install /path/to/wheelfile.whl
  ```