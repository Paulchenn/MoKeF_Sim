# MoKeF - Simulation

**modelfilter_ibvs** is a lightweight Python framework and simulator for
image‑based visual servoing (IBVS) with homography estimation.  The
project bundles a 6‑DOF robot/camera simulation, a simple Qt‑based GUI,
a collection of feature extractors and tracking filters, and an extensible
control library.  It is intended for research, teaching and rapid
prototyping of IBVS algorithms.

---

## 🚀 Features

- **IBVS controller** implementing classical interaction‑matrix based
  image‑space servo.
- **Homography tracking filters**
  - complementary Lie‑group filter (`HomographyFilter`)
  - extended Kalman filter (`HomographyEKF`) with gating and noise
    modelling.
- **Feature extraction and matching**
  - ORB (fallback / real‑time)
  - XFeat CNN‑based extractor (via [`accelerated_features` torch hub
    module](https://github.com/verlab/accelerated_features)).
- **6‑DOF robot and camera simulator** for closed‑loop experimentation
  (`sim/robot_sim.py`, `sim/camera_sim.py`).
- **Qt GUI** for interactive simulation and parameter tuning
  (`sim/gui_main.py`).
- Minimal ROS node wrappers for control and filtering under `ros_nodes/`.
- A small collection of utilities for kinematics, math, and noise
  injection.

---

## 🛠️ Getting started

### Prerequisites

- **Python 3.10** (project was developed with 3.10; newer versions should
  work but have not been tested).
- `pip` and a C‑compiler (required by `opencv-python`).
- The packages listed in `requirements.txt`:
  ```text
  numpy==1.26.4
  scipy==1.11.4
  matplotlib
  PyQt5           # GUI
  roboticstoolbox-python  # kinematics/math helpers
  opencv-python   # image processing
  spatialmath-python # homogeneous transforms
  ```

> **Tip:** create and activate a venv before installing:
> ```bash
> python -m venv .venv
> source .venv/bin/activate
> pip install -r requirements.txt
> ```

### Installation

From the project root (where `setup.py` lives) you can install in editable
mode:

```bash
pip install -e .
```

This makes the `modelfilter_ibvs` package importable anywhere on your
system.

### Running the simulator

Launch the GUI from the root directory:

```bash
python -m modelfilter_ibvs.sim.gui_main
```

Use the controls to:

1. Select a feature extractor and a tracking filter.
2. Capture the current view as the desired reference.
3. Start/stop servoing; noise and glare can be simulated.
4. Manually drive the six robot joints.

The left‑hand side of the window displays the 3‑D robot and the camera
image with feature matches and estimated homography.

### Using the library programmatically

All core components are designed to work independently of the GUI. A
typical workflow looks like:

```python
from modelfilter_ibvs.core.ibvs import IBVSController
from modelfilter_ibvs.core.filter_EKF import HomographyEKF
from modelfilter_ibvs.core.feature_extractor import XFeatExtractor

cam_K = np.eye(3)  # your intrinsic matrix
controller = IBVSController(cam_K)
filter = HomographyEKF(cam_K)
extractor = XFeatExtractor()

# set reference image, run matching, compute velocity, predict/update filter, etc.
```

See the `accelerated_features/minimal_example.py` and the notebooks under
`modelfilter_ibvs/accelerated_features/notebooks` for more usage examples.

---

## 🗂️ Project structure

```
modelfilter_ibvs/             # top‑level package
├── core/                     # algorithmic building blocks
│   ├── ibvs.py               # interaction matrix controller
│   ├── filter.py             # Lie‑group homography filter(s)
│   ├── filter_EKF.py         # extended Kalman filter variant
│   ├── feature_extractor.py  # ORB/XFeat extraction & matching
│   ├── kinematics.py         # simple robot kinematics
│   ├── math_utils.py         # helpers (skew, etc.)
│   └── ...                   # other utilities
├── sim/                      # simulator and GUI
│   ├── robot_sim.py
│   ├── camera_sim.py
│   └── gui_main.py
├── ros_nodes/                # ROS wrapper nodes (control, filter)
├── accelerated_features/     # vendor code for XFeat (submodule)
├── notebooks/                # Jupyter examples for development
├── weights/                  # pretrained models (XFeat, etc.)
└── ...
```

### Key modules

| Module                         | Purpose                                      |
|-------------------------------|----------------------------------------------|
| `core.ibvs`                   | Basic IBVS control laws                      |
| `core.filter`                 | Homography filtering on SE(3)                |
| `core.filter_EKF`             | EKF estimator for homography                |
| `core.feature_extractor`      | Abstraction over ORB/XFeat feature matching  |
| `sim.robot_sim`               | Robot joint and camera simulation model     |
| `sim.camera_sim`              | Simple pinhole camera and image rendering   |

---

## 📘 Examples

- **Minimal Python script:**
  `accelerated_features/minimal_example.py` shows basic usage of the
  XFeat extractor.
- **Jupyter notebooks:** several demo notebooks under
  `modelfilter_ibvs/accelerated_features/notebooks` illustrate feature
  matching and model loading.
- **GUI demo:** run `gui_main.py` and explore the different extractor and
  filter combinations.

---

## ⚙ Configuration

- Intrinsic camera parameters are set in `sim/robot_sim.py`; you can
  modify them or load from calibration files.
- Filter and extractor parameters (number of features, noise levels,
  alpha gain, EKF covariance, etc.) are exposed in the GUI and can be
  changed dynamically.

---

## 🧪 Testing

At present the repository does not include a formal test suite.  You can
manually exercise components by running the simulator or importing
individual classes in an interactive session.

---

## 🤝 Contributing

Contributions are welcome!  Please follow these guidelines:

1. Fork the repository and create a feature branch.
2. Write clear, well‑documented code and add comments where helpful.
3. If you add new functionality, include examples or a new notebook.
4. Submit a pull request describing your changes.

Keep the code style consistent with existing modules (PEP 8, simple
function/class names, in‑line German comments may remain for context but
prefer English for public APIs).

---

## 📄 License

This project currently does not specify a license.  Add a `LICENSE`
file or update `setup.py` if you intend to release the code under an
open‑source license such as MIT, BSD or Apache.

---

## 🧾 Acknowledgments

- The **XFeat** model is loaded via the
  [accelerated_features](https://github.com/verlab/accelerated_features)
  repository.
- PyQt5, OpenCV and Robotics Toolbox provide foundational capabilities.

---

```text
Author: Paul Glaser – February 2026
```
