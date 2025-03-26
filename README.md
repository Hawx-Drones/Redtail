# redtail.ai

A comprehensive autonomous drone navigation system that integrates reinforcement learning, object detection, SLAM, and path planning.

> ⚠️ This repository is part of the open-source `redtail.ai` training stack.
> It does **not** include trained model weights, proprietary SDKs, or secure control interfaces used in commercial deployments.

## Architecture Overview

This system combines four key AI technologies to enable fully autonomous drone flight:

1. **Reinforcement Learning (Brain) - Stable-Baselines3**
   - Decision-making system that learns optimal flight behavior
   - Processes sensor data to adjust flight decisions in real-time
   - Implements the PPO algorithm for efficient training

2. **Object Detection (Vision) - Ultralytics YOLO**
   - Real-time obstacle detection
   - Identifies objects in the drone's environment
   - Feeds obstacle data to the RL model for collision avoidance

3. **Mapping & Localization (Positioning) - RTAB-Map**
   - Builds real-time 3D maps using SLAM
   - Tracks drone position in the environment
   - Provides spatial awareness for navigation

4. **Path Planning (Motion Planning) - OMPL**
   - Plans collision-free routes in real time
   - Works with obstacle detection to find safe paths
   - Provides planned routes to the RL model

## Prerequisites

- PX4 Autopilot (SITL)
- Gazebo Harmonic
- QGroundControl
- Python 3.8+
- CUDA-compatible GPU (recommended for YOLO)

## Installation

1. **Clone the repositories**:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install MAVSDK Python
pip install mavsdk
```

2. **Install external dependencies**:

```bash
# Install YOLO
pip install ultralytics

# Install Stable-Baselines3
pip install stable-baselines3[extra]

# Install RTAB-Map
# Follow instructions at: https://github.com/introlab/rtabmap

# Install OMPL
# Follow instructions at: https://ompl.kavrakilab.org/installation.html

# Set up PX4 SITL
git clone https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot
make make px4_sitl gz_x500
```

## Project Structure

```
redtail.ai/
├── drone_env.py            # RL environment interfacing with PX4
├── yolo_detector.py        # YOLO object detection integration
├── rtabmap_interface.py    # RTAB-Map SLAM integration
├── ompl_integration.py     # OMPL path planning integration
├── integration_script.py   # Main system integration script
├── train_drone.py          # Training script for RL model
├── models/                 # Saved RL models directory
├── logs/                   # Training logs directory
└── tensorboard/           # Tensorboard logs for training visualization
```

## Usage

### Training the RL Model

Run the training script to train the reinforcement learning model:

```bash
# Start training with default parameters
python train_drone.py

# Customize training parameters
python train_drone.py --timesteps 200000 --connection udp://:14540
```

Training parameters:
- `--timesteps`: Number of training steps (default: 100000)
- `--connection`: MAVSDK connection string (default: udp://:14540)
- `--px4-dir`: Custom PX4 directory (default: ~/PX4-Autopilot)
- `--no-gazebo`: Don't start Gazebo (if it's already running)
- `--eval`: Evaluate model after training

The training process will save model checkpoints in the `models/` directory.

### Deploying the Autonomous System

After training, you can deploy the system in different modes:

#### 1. RL Mode (using the trained model)

```bash
python integration_script.py --deploy --model models/best_model.zip
```

#### 2. Path Planning Mode (without RL)

```bash
python integration_script.py --plan
```

### System Configuration

You can modify the following files to customize system behavior:

- `drone_env.py`: Change reward functions, observation spaces, or action spaces
- `yolo_detector.py`: Adjust detection confidence or object classes of interest
- `rtabmap_interface.py`: Modify mapping parameters
- `ompl_integration.py`: Change path planning algorithms or parameters

## Working with Jetson Orin Nano

To deploy on Jetson Orin Nano:

1. Install the required libraries (YOLO, RTAB-Map, OMPL, etc.) on the Jetson
2. Install MAVSDK and your hawxsdk wrapper
3. Copy the trained model to the Jetson
4. Run the integration script in deployment mode:

```bash
python integration_script.py --deploy --model models/best_model.zip --connection your_mavsdk_connection
```

## Visualization

The system provides several visualization options:

1. **Training Progress**: View with Tensorboard
   ```bash
   tensorboard --logdir=./tensorboard
   ```

2. **Path Planning**: The planned path is visualized in QGC if enabled

3. **SLAM Mapping**: RTAB-Map provides its own visualization tools
   ```bash
   rtabmap path/to/saved/database.db
   ```

## Customization

### Adding Custom Obstacles for Training

You can add custom obstacles in the `DroneEnv` class:

```python
# In drone_env.py
def _check_collision(self, position):
    # Define obstacles
    obstacles = [
        {"center": np.array([20.0, 20.0, -5.0]), "radius": 5.0},
        # Add more obstacles here
    ]
    # ...
```

### Implementing Custom Reward Functions

Modify the reward function in `DroneEnv` to prioritize different behaviors:

```python
# In drone_env.py
def _compute_reward(self, observation):
    # Example: reward for moving toward goal while avoiding obstacles
    position = observation[:3]
    distance_to_goal = np.linalg.norm(position - self.goal_position)
    distance_reward = -distance_to_goal / 10.0
    
    # Add your custom rewards here
    # ...
    
    return distance_reward + your_custom_reward
```

## Troubleshooting

### PX4 Connection Issues

If you have trouble connecting to PX4:

```bash
# Check if SITL is running
ps aux | grep px4

# Verify MAVLink ports
netstat -tuln | grep 14540

# Restart PX4 SITL
cd ~/PX4-Autopilot
make px4_sitl gazebo-classic_x500
```

### YOLO Detection Issues

For YOLO detection problems:

```bash
# Test YOLO separately
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); results = model('path/to/test/image.jpg', show=True)"
```

## Project Scope

This repository includes:

- Full training environment and reinforcement learning pipeline for autonomous drone navigation
- Modular interfaces for YOLO-based object detection, RTAB-Map SLAM, and OMPL path planning
- Simulation-ready training and deployment scripts using PX4 SITL and Gazebo

**Not included**:

- Proprietary SDKs such as `HawxSDK` used for production drone deployments
- Trained model weights (`models/` is excluded from this repo)
- Live flight control integrations or secure communication layers

This open-source release is intended for **research, reproducibility, and community exploration** of drone autonomy.

If you're a public safety agency or defense partner interested in full-system access, please contact us via [hawx.us](https://hawx.us).


## License

The contents of this repository are licensed under the MIT License.
Trained models, SDKs, and production integration are not included and are licensed separately for commercial use.

## Acknowledgments

- PX4 Autopilot team
- Ultralytics YOLO
- RTAB-Map developers
- OMPL contributors
- Stable-Baselines3 team
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- PX4 Documentation: https://docs.px4.io/
- MAVSDK Python Documentation: https://mavsdk-python.readthedocs.io/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- Ultralytics YOLO: https://docs.ultralytics.com/
- RTAB-Map: http://introlab.github.io/rtabmap/
- OMPL: https://ompl.kavrakilab.org/