<img src="https://github.com/user-attachments/assets/56821f4d-f307-40ba-9d12-f8fca06c186e" p align="center" style="Width:50% ; height:auto;">

# Cardillo
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![License](https://img.shields.io/badge/code%20standard-PEP8-black)](https://www.python.org/dev/peps/pep-0008/)

Cardillo is an open-source simulation framework for flexible multi-body systems with frictional contacts and impacts developed by teams from the University of Stuttgart, the Eindhoven University of Technology (TU/e), and the Friedrich-Alexander-Universität Erlangen-Nuremberg (FAU). Thanks to its modular structure, Cardillo is ideal for developing and testing new algorithms and models, and can be used for the simulation of real-world systems.

-Maybe some information about the reason behind Cardillo, if needed-

Enjoy how our Cardillo can be implemented in real-life scenarios by looking at the following examples.

## Examples 
|Rockfall|Two-mass Oscillator|
|--- |---| 
| ![Rockfall](https://github.com/user-attachments/assets/412621b0-bbf5-4213-8327-e983b2430283)|![Two_mass_oscillator](https://github.com/user-attachments/assets/0e3911d0-b689-4f38-85cc-8c14513fbe8c)  
|Spinning Top| Multiple Balls|
| ![Spinning_top](https://github.com/user-attachments/assets/88a9b65d-272f-432b-ae7b-925e3b00ae0f) |![multiple_balls](https://github.com/user-attachments/assets/899bd885-e3d0-460b-a133-174357e2f841) 


## Requirements

## Tutorials

|Name of the tutorial                     .|Use cases|
|  :---:  | ---:  |
|*__Bouncing Ball__* |[Bouncing Ball](examples/bouncing_ball/bouncing_ball.py)|
  |![Bouncing_ball](https://github.com/user-attachments/assets/135f783d-da23-4b80-8a36-9e1c8edfd8de) | This tutorial encompasses the use of Moreau solver for contact between rigid bodies. You can use this example to build any contact problems between a movable and an immovable object combination, for example throwing a frisbee against a couple of walls, books falling from table, closing a cabinet door, etc.,|
|*__Double Pendulum__*|[Double Pendulum](examples/double_pendulum/double_pendulum.py)|
|  ![Double_pendulum](https://github.com/user-attachments/assets/96f98dae-b50c-4b4a-8f6b-fe65e00d2b71)          |This double Pendulum is a classic example of multiple bodies moving relative to each other. To do so, the relative position is derived from the math module and it uses scipy´s solve_ivp and scipy_dae methods that handle multi-body problems, where the constraints are handled using Lagrange Multipliers and the differential-algebraic equations are solved respectively|
|*__Dzhanibekov Effect__*|[Dzhanibekov Effect](examples/dzhanibekov_effect)|
| ![Dzhanibkov_effect](https://github.com/user-attachments/assets/2beb0a9b-fe1c-43b7-9c08-48128cc478db)          |The Dzhanibekov Effect is akin to a child spinning a top, and we replicated this phenomenon by exploring gyroscopic forces in a space-like environment. We used the RATTLE solver to manage the position and velocity of systems with holonomic constraints, ensuring that these constraints are maintained throughout the simulation. This experiment illustrates how RATTLE accurately models the dynamics of rotating bodies, enabling the observation of complex behaviors such as the flipping motion characteristic of the Dzhanibekov effect.  |
|*__Rolling Disk__*|[Rolling disk](examples/rolling_disc)|
|  ![Rolling_disks](https://github.com/user-attachments/assets/8a097f6c-097b-4f07-8027-b0ced33a0b97)             |This is a common experience for kids playing with a coin. This experiment interestingly combines key methods from previous examples, such as the reference plane, RATTLE solver, and Scipy's solve_ivp. The reference plane defines the spatial framework for simulating the rigid disc's motion, ensuring impenetrability; the RATTLE solver maintains the nonholonomic rolling condition by preventing sliding at the velocity level; and solve_ivp integrates the equations of motion, ensuring that constraints are satisfied throughout the simulation. |


## Getting Started
### Requirements
* Python 3.x (tested on Python 3.10 for Ubuntu 22.04.2 LTS)
* We recommend creating a virtual environment to install the required packages, ensuring it doesn't interfere with your other projects.
  This also makes it easier to install future updates from our side, improving performance through package and program updates.
* Please follow the steps to create a virtual environment or please visit "https://docs.python.org/3/library/venv.html"

## Installation
To install the package, clone or download the current repository, open a console, navigate to the root folder of the project and run `pip install .`.
### Installation
To install the package, clone or download the current repository, open a console, navigate to the project's root folder, and run `pip install .`

```bash
git clone https://github.com/cardilloproject/cardillo.git
cd cardillo
pip install .
```
## For Students and Developers 
To know a bit more about Project Workflow, and some installation steps please refer: https://github.com/cardilloproject/project_template

# Team
Team Member list, if needed
