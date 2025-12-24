# LeNewton

LeNewton is a simulation environment for the SO100 robot arm, built on top of the Newton physics engine and NVIDIA Warp. It provides a high-performance, differentiable simulation platform for robotic manipulation tasks.


## Installation

### Prerequisites

- Python 3.11
- CUDA-capable GPU (recommended for Warp/Newton)

### Install from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/makolon/lenewton.git
   cd lenewton
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

## Usage

### Running Examples

You can find example scripts in the `examples/` directory. To run the basic environment creation and block stacking task example:

```bash
python examples/01_create_env.py
```

This script demonstrates:
1. Creating a `BlockStackTask`.
2. Initializing the `LeNewtonEnv` with the SO100 robot.
3. Setting up the viewer and video recording.

### Creating a Custom Task

Tasks are defined by inheriting from `LeNewtonTask`. See `src/lenewton/tasks/block_stacking.py` for a reference implementation.

```python
from lenewton.env import LeNewtonEnv
from lenewton.tasks import BlockStackTask

# Define the task
task = BlockStackTask(
    task_name="BlockStack",
    goal_str="Stack the blocks",
    max_steps=500
)

# Initialize the environment
env = LeNewtonEnv(
    task=task,
    render_mode="rgb_array",
    use_viewer=True
)

# Simulation loop
obs = env.reset()
# ... action loop ...
```

## Project Structure

```
lenewton/
├── examples/           # Example scripts
├── src/
│   └── lenewton/
│       ├── assets/     # Robot URDFs, meshes, and scene XMLs
│       ├── tasks/      # Task definitions (e.g., BlockStackTask)
│       ├── env.py      # Main environment class
│       └── task.py     # Base task class
├── pyproject.toml      # Project configuration and dependencies
└── README.md
```

## License

This project is licensed under the terms of the included [LICENSE](LICENSE) file.