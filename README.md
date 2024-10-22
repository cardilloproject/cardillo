<img src="https://github.com/user-attachments/assets/39ef1914-7d52-40ce-87fb-742f51530121" alt="Alt Text" style="Width:50% ; height:auto;">

# Cardillo
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![License](https://img.shields.io/badge/code%20standard-PEP8-black)](https://www.python.org/dev/peps/pep-0008/)

Cardillo is an open-source simulation framework for flexible multi-body systems with frictional contacts and impacts. Teams from the University of Stuttgart, the Eindhoven University of Technology (TU/e), and the Friedrich-Alexander-Universität Erlangen-Nuremberg (FAU) developed the software. It has been used to solve real-world problems involving basic motions such as bouncing balls, double pendulums, and friction belts. Cardillo is a Python framework that applies the fundamental principle of modularity to multi-body dynamics problems.

-Maybe some information about the reason behind Cardillo, if needed-

Enjoy how our Cardillo can be implemented in real-life scenarios by taking a look at the following examples.

##  Examples
|Name|Description|
|-|-|
|![Bouncing Ball](https://github.com/user-attachments/assets/4f337864-ddb6-490f-b7d5-47fcc55f58af) | The simulation of the bouncing ball here displayed as a picture is an attempt to implement Moreau´s time-stepping scheme, which is generally used in scenarios involving contact, impact and friction; which can be very well studied by a simple experiment of dropping a ball on the ground|
|*__Bouncing Ball__* |[Bouncing Ball](examples/bouncing_ball/bouncing_ball.py)|
|![Double Pendulum](https://github.com/user-attachments/assets/3ea744d4-b732-4a22-8038-cb67d96da227) |This double Pendulum is a classic example of multiple bodies moving relative to each other. To do so, the relative position is derived from the math module and it uses scipy´s solve_ivp and scipy_dae methods that handle multi-body problems, where the constraints are handled using Lagrange Multipliers and the differential-algebraic equations are solved respectively|
|*__Double Pendulum__*|[Double Pendulum](examples/double_pendulum/double_pendulum.py)|
|![Dzhanibekov Effect](https://github.com/user-attachments/assets/149fd423-bcf7-4c1c-ab30-d5ca0cc1ea2a) |The Dzhanibekov Effect is akin to a child spinning a top, and we replicated this phenomenon by exploring gyroscopic forces in a space-like environment. We used the RATTLE solver to manage the position and velocity of systems with holonomic constraints, ensuring that these constraints are maintained throughout the simulation. This experiment illustrates how RATTLE accurately models the dynamics of rotating bodies, enabling the observation of complex behaviors such as the flipping motion characteristic of the Dzhanibekov effect.  |
|*__Dzhanibekov Effect__*|[Dzhanibekov Effect](examples/dzhanibekov_effect)|
|![Rolling Disk](https://github.com/user-attachments/assets/35969216-84f7-4174-9358-1c894c322838) |This is a common experience for kids playing with a coin. This experiment interestingly combines key methods from previous examples, such as the reference plane, RATTLE solver, and Scipy's solve_ivp. The reference plane defines the spatial framework for simulating the rigid disc's motion, ensuring impenetrability; the RATTLE solver maintains the nonholonomic rolling condition by preventing sliding at the velocity level; and solve_ivp integrates the equations of motion, ensuring that constraints are satisfied throughout the simulation. |
|*__Rolling Disk__*|[Rolling disk](examples/rolling_disc)|

A ton of other examples, solvers and system building blocks have been developed and a whole lot is still in progress, install the Cardillo by the steps mentioned below and utilise our creation to the fullest.

Enjoy the constraint-rich, frictioned, non-linear trajectory!!!

## Getting Started
### Requirements
* Python 3.x (tested on Python 3.10 for Ubuntu 22.04.2 LTS)
* We recommend creating a virtual environment to install the required packages, ensuring it doesn't interfere with your other projects.
  This also makes it easier to install future updates from our side, improving performance through package and program updates.
* Please follow the steps to create a virtual environment or please visit "https://docs.python.org/3/library/venv.html"

### Installation
To install the package, clone or download the current repository, open a console, navigate to the root folder of the project and run `pip install .`.

```bash
git clone https://github.com/cardilloproject/cardillo.git
cd cardillo
pip install .
```
* It is also recommended to have an OpenGL library, if not already installed, follow the steps mentioned below (Linux)
* For Ubuntu or Debian-based systems, run:

```bash
Copy code
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
```
* Verify installation:
Once the package is installed, try running the the script of our very first example that simulates a ball bouncing on a flat surface

```bash
Copy code
python examples/bouncing_ball/bouncing_ball.py
```

### Virtual environment considerations:

* Please ensure your virtual environment is active when installing system-wide dependencies like OpenGL. Virtual environments typically rely on system libraries for dependencies like libGL.so.1.

# Team
Team Member list, if needed
