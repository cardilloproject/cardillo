from setuptools import setup, find_packages

name = "cardillo_urdf"
version = "0.0.1"
author = ""
author_email = "giuseppe.capobianco@fau.de"
description = ""
long_description = ""

setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    install_requires=[
        "cardillo",
        # "cardillo @ git+https://github.tik.uni-stuttgart.de/inm-cardillo/cardillo.git@cardillo-core#egg=cardillo"
        "urchin", # maintained fork of urdfpy, see https://github.com/fishbotics/urchin
        "networkx>=3",
        "cachetools", #TODO: remove if rigid_body_rel_kinematics is moved to cardillo-core
        "pyglet<2",
    ],
    python_requires=">=3.8",
    packages=find_packages(),
)
