# Autonomous navigation in Webots

## Overview

This project implements an autonomous navigation pipeline for a mobile robot in a Webots simulation.

The robot explores the environment, builds an occupancy grid map from **LiDAR**, detects landmarks and hazards with **camera-based color segmentation**, and plans routes using global **A\*** with local **DWA path following**.

## Requirements

- Python 3.x
- Webots
- Python packages listed in [requirements.txt](requirements.txt)

## Installation

Install the Python dependencies from [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

## Webots setup (selecting the controller)

1. Open your world in Webots.
2. In the robot node properties, set **controller** to `main`.
3. Ensure the controller directory points to [src/controllers/main](src/controllers/main) and the entry file is [src/controllers/main/main.py](src/controllers/main/main.py).
4. Start the simulation.
