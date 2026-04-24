# Cardillo

<p align="center">
  <img src="https://github.com/user-attachments/assets/56821f4d-f307-40ba-9d12-f8fca06c186e" alt="Cardillo banner" width="70%">
</p>

<p align="center">
  <em>Open-source Python framework for flexible multi-body systems with frictional contacts and impacts</em>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/BSD-3-Clause">
    <img src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg" alt="License: BSD 3-Clause">
  </a>
  <a href="https://www.python.org/dev/peps/pep-0008/">
    <img src="https://img.shields.io/badge/code%20standard-PEP8-black" alt="Code Standard: PEP8">
  </a>
  <a href="https://github.com/cardilloproject/cardillo/actions/workflows/main.yml">
    <img src="https://github.com/cardilloproject/cardillo/actions/workflows/main.yml/badge.svg" alt="Tests">
  </a>
  <img src="https://img.shields.io/badge/python-≥3.10-blue.svg" alt="Python ≥ 3.10">
</p>

---

**Cardillo** is a research-grade Python simulation framework that brings together rigid and flexible body dynamics, frictional contact mechanics, and impact modelling in a single, composable package. It is developed by teams from the [University of Stuttgart](https://www.uni-stuttgart.de), the [Eindhoven University of Technology (TU/e)](https://www.tue.nl) and the [Friedrich-Alexander-Universität Erlangen-Nuremberg (FAU)](https://www.fau.de).

---

## Table of Contents

- [Key Features](#key-features)
- [Simulation Gallery](#simulation-gallery)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Framework Overview](#framework-overview)
- [Examples](#examples)
- [Contributing](#contributing)
- [Authors](#authors)
- [License](#license)

---

## Key Features

- 🔩 **Flexible bodies** — Geometrically exact Cosserat rod theory for beams with bending, torsion, extension and shear
- 🧲 **Frictional contact & impact** — Sphere-to-plane and sphere-to-sphere contact with Coulomb friction and restitution coefficients
- 🔗 **Holonomic constraints** — Revolute, prismatic, spherical, cylindrical, fixed-distance and rigid-connection joints
- ⚙️ **Control & actuation** — PD/PID controllers, motor models and optimal-control examples
- 📐 **Multiple integrators** — Nonsmooth (Moreau, Dual Störmer-Verlet), DAE (RATTLE, SciPy DAE) and ODE (Backward Euler, SciPy IVP) solvers, plus a nonlinear statics RIKS solver
- 🤖 **Robot integration** — URDF parser with ready-made examples for the Franka Emika Panda arm and the Unitree Go1 quadruped
- 📊 **Visualization & export** — VTK rendering, STL mesh export, trimesh integration and real-time animation
- 🛠️ **Composable API** — Build any system by assembling bodies, constraints, forces and contacts around a sparse-matrix `System` core

---

## Simulation Gallery

<table>
  <tr>
    <td align="center"><strong>Rockfall</strong></td>
    <td align="center"><strong>Two-mass Oscillator</strong></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/412621b0-bbf5-4213-8327-e983b2430283" alt="Rockfall simulation" width="100%"></td>
    <td><img src="https://github.com/user-attachments/assets/0e3911d0-b689-4f38-85cc-8c14513fbe8c" alt="Two-mass oscillator simulation" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><strong>Spinning Top</strong></td>
    <td align="center"><strong>Multiple Balls</strong></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/88a9b65d-272f-432b-ae7b-925e3b00ae0f" alt="Spinning top simulation" width="100%"></td>
    <td><img src="https://github.com/user-attachments/assets/899bd885-e3d0-460b-a133-174357e2f841" alt="Multiple balls simulation" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><strong>Bouncing Ball</strong></td>
    <td align="center"><strong>Double Pendulum</strong></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/135f783d-da23-4b80-8a36-9e1c8edfd8de" alt="Bouncing ball simulation" width="100%"></td>
    <td><img src="https://github.com/user-attachments/assets/96f98dae-b50c-4b4a-8f6b-fe65e00d2b71" alt="Double pendulum simulation" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><strong>Dzhanibekov Effect</strong></td>
    <td align="center"><strong>Rolling Disc</strong></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/2beb0a9b-fe1c-43b7-9c08-48128cc478db" alt="Dzhanibekov effect simulation" width="100%"></td>
    <td><img src="https://github.com/user-attachments/assets/8a097f6c-097b-4f07-8027-b0ced33a0b97" alt="Rolling disc simulation" width="100%"></td>
  </tr>
</table>

---

## Installation

**Requirements:** Python ≥ 3.10

Clone the repository and install with pip:

```bash
git clone https://github.com/cardilloproject/cardillo.git
cd cardillo
pip install .
```

> **Tip:** Use a virtual environment (`python -m venv .venv && source .venv/bin/activate`) to keep dependencies isolated.

> **Note:** 3-D visualization uses [VTK](https://vtk.org). On headless Linux systems (e.g. CI servers) you may need a virtual display (e.g. `Xvfb`) or install additional system packages. All other functionality works without a display.

---

## Quick Start

The snippet below simulates a 3-D sphere bouncing off a flat plane with Coulomb friction and coefficient of restitution — all in under 30 lines:

```python
import numpy as np
from cardillo import System
from cardillo.discrete import RigidBody, Sphere, Frame
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane
from cardillo.solver import Moreau

# Build system
system = System()

# Geometry and initial conditions
radius = 0.05
r_OC0 = np.array([-0.75, 0.0, 8 * radius])
v_C0  = np.array([1.0,   0.0, 0.0])

# Create sphere
ball = Sphere(RigidBody)(
    radius=radius, density=1.0, subdivisions=3,
    q0=RigidBody.pose2q(r_OC0, np.eye(3)),
    u0=np.hstack([v_C0, np.zeros(3)]),
    name="ball",
)

# Ground plane and contact model (e_N = restitution, mu = friction)
ground = Frame(name="ground")
contact = Sphere2Plane(ball, ground, mu=0.5, e_N=0.75, e_F=0.0, name="contact")

# Gravity
gravity = Force(np.array([0, 0, -9.81]) * ball.mass, ball, name="gravity")

system.add(ball, ground, contact, gravity)

# Solve and inspect
sol = Moreau(system).solve(t1=3.0, dt=1e-3)
print("Final height:", sol.q[-1][2])
```

More complete, runnable examples live in the [`examples/`](examples/) directory.

---

## Framework Overview

Cardillo is organised into focused sub-packages that can be mixed and matched:

| Sub-package | Description |
|---|---|
| `cardillo.system` | Sparse-matrix `System` class — the central assembly engine |
| `cardillo.discrete` | Rigid bodies, point masses, frames and geometric shapes (`Sphere`, `Box`, `Meshed`) |
| `cardillo.rods` | Geometrically exact Cosserat rods for flexible beams |
| `cardillo.contacts` | Contact mechanics: `Sphere2Plane`, `Sphere2Sphere` |
| `cardillo.constraints` | Joint library: `Revolute`, `Prismatic`, `Spherical`, `Cylindrical`, … |
| `cardillo.forces` | Applied forces and moments |
| `cardillo.force_laws` | Constitutive models: springs, Kelvin-Voigt and Maxwell dampers |
| `cardillo.actuators` | PD/PID controllers and motor models |
| `cardillo.solver` | Time-integration schemes: `Moreau`, `DualStormerVerlet`, `RATTLE`, `ScipyDAE`, `ScipyIVP`, … |
| `cardillo.urdf` | URDF parser — turn robot descriptions into Cardillo systems |
| `cardillo.visualization` | VTK rendering, animation and mesh export |
| `cardillo.utility` | Sensors, convergence analysis, Bézier curves, state I/O |
| `cardillo.math` | Rotation algebra, proximal operators and numerical helpers |

---

## Examples

The [`examples/`](examples/) directory contains 18 self-contained simulations covering a wide range of application areas:

| Example | Highlights |
|---|---|
| [`bouncing_ball`](examples/bouncing_ball/) | Sphere with friction bouncing on a plane; restitution |
| [`double_pendulum`](examples/double_pendulum/) | 2-link rigid-body pendulum with STL mesh geometries |
| [`rolling_disc`](examples/rolling_disc/) | Disc rolling on a plane with non-holonomic constraints |
| [`rod2plane`](examples/rod2plane/) | Flexible Cosserat beam contacting a rigid plane |
| [`elastic_chain_pendulum`](examples/elastic_chain_pendulum/) | 20-particle elastic chain with spring-damper elements |
| [`top`](examples/top/) | Spinning top under gravity — gyroscopic precession |
| [`dzhanibekov_effect`](examples/dzhanibekov_effect/) | Tennis-racket / intermediate-axis effect |
| [`woodpecker_toy`](examples/woodpecker_toy/) | Woodpecker toy as an impulsive multi-body mechanism |
| [`inverted_pendulum_PID`](examples/inverted_pendulum_PID/) | Stabilised inverted pendulum with PID control |
| [`inverted_pendulum_OC`](examples/inverted_pendulum_OC/) | Optimal-control formulation for pendulum swing-up |
| [`urdf_panda`](examples/urdf_panda/) | Franka Emika Panda robot arm loaded from URDF |
| [`urdf_unitree_go1`](examples/urdf_unitree_go1/) | Unitree Go1 quadruped loaded from URDF |
| [`Herrmann2025_mixed_Cosserat_rod`](examples/Herrmann2025_mixed_Cosserat_rod/) | Mixed-formulation Cosserat rod (publication examples) |
| … and more | See [`examples/`](examples/) for the full list |

Run any example directly:

```bash
python examples/bouncing_ball/bouncing_ball.py
```

---

## Contributing

Contributions are very welcome! To get started:

1. Fork the repository and create a feature branch.
2. Install the package along with the development tools (`black` for formatting and `pytest` for testing): `pip install . && pip install black pytest`.
3. Follow the **PEP 8** code style — the CI will check formatting with [black](https://github.com/psf/black).
4. Add or update tests in `test/` and make sure `pytest ./test/` passes.
5. Open a pull request describing your changes.

---

## Authors

Cardillo is developed and maintained by:

| Name | Affiliation |
|---|---|
| **[Jonas Breuling](https://github.com/JonasBreuling)** | [University of Stuttgart](https://www.inm.uni-stuttgart.de/institut/mitarbeiter/Harsch/) |
| **[Giuseppe Capobianco](https://github.com/capobiag)** | [Friedrich-Alexander-Universität Erlangen-Nuremberg (FAU)](https://www.ltd.tf.fau.de/faudir/giuseppe-capobianco/) |
| **[Lisa Eberhardt](https://github.com/lisaeb)** | [University of Stuttgart](https://www.inm.uni-stuttgart.de/institut/mitarbeiter/Eberhardt/) |
| **[Simon R. Eugster](https://github.com/simonreugster)** | [Eindhoven University of Technology (TU/e)](https://www.tue.nl/en/research/researchers/simon-eugster) |

---

## License

Cardillo is distributed under the **BSD 3-Clause License**. See [`LICENSE.txt`](LICENSE.txt) for details.
