The tracking on a setup with multiple GPUs can be done with `toolbox.track_multigpu()`. This function divides an `xtrack.Particles` object between the available OpenCL GPU devices. Each worker constructs its own line and tracking context.

## Example usage

```python
from toolbox import track_multigpu
import xtrack as xt


def line_constructor() -> xt.Line:
	reference_line = xt.Line(
		elements = [
			xt.Drift(length = 2.), 
			xt.Multipole(knl = [0, 0.5], ksl = [0, 0]),
			xt.Drift(length = 1.),
			xt.Multipole(knl = [0, -0.5], ksl = [0, 0])],
		element_names = ['drift_0', 'quad_0', 'drift_1', 'quad_1']
		)

	reference_line.particle_ref = xt.Particles(
		p0c = 6500e9,
		q0 = 1, 
		mass0 = xt.PROTON_MASS_EV
	)

	return reference_line

if __name__ == "__main__":

	line = line_constructor()

	p = line.build_particles(
		x = np.random.uniform(-1e-3, 1e-3, 10_000),
		px = np.random.uniform(-1e-5, 1e-5, 10_000),
		method = '4d',
	)

	tracked_particles = track_multigpu(
    p,
    line_constructor = line_constructor,
    num_turns = 100_000,
    num_gpus = 2,
    verbose = 1,
)
```
**The script requires `line_constructor` to be a top-level function, because it is pickled and sent to the workers. The `if __name__ == "__main__":` guard is required to avoid recursive spawning of subprocesses.**

## Saving phase-space snapshots
It is possible to record the phase-space snapshots of the tracked particles by providing `record_every` and `monitor_output_directory`.  
`xtrack.ParticlesMonitor` will record the snapshots at the specified intervals. The snapshots are written in HDF5 format and are exported to the specified folder after the tracking is finished.

```python
tracked_particles = track_multigpu(
    p,
    line_constructor = line_constructor,
    num_turns = 100_000,
    num_gpus = 2,
    verbose = 1,
    record_every = 1000,
    monitor_output_directory = "./monitor_snapshots"
)
```

## Tracking Checkpoints
It is possible to save checkpoints during the tracking by providing `checkpoint_every` and `checkpoint_output_directory`.  
The `xt.Particles` object will be saved every `checkpoint_every` turns and exported to the specified directory during the tracking.  
This allows to resume the tracking from the last checkpoint in case of an interruption.

```python
tracked_particles = track_multigpu(
    p,
    line_constructor = line_constructor,
    num_turns = 100_000,
    num_gpus = 2,
    verbose = 1,
    checkpoint_every = 1000,
    checkpoint_output_directory = "./checkpoint_snapshots"
)
```

## Running on GSI HPC cluster
There is a HYDRA cluster at GSI with AMD and NVIDIA GPUs. To use one needs to write a Slurm script. An example to use AMD GPUs:
```bash
#!/bin/bash

#SBATCH --partition=amd_gpu
#SBATCH --gres=gpu:8
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=16
#SBATCH --job-name=SX_KO_He
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err	

set -euo pipefail

singularity exec xsuite_image.sif python3 tracking.py
```
The script that does the tracking is `"tracking.py"`. A proper image (`.sif`) with all dependencies has to be used. There are prebuilt images with all necessary dependencies and `xsuite` and `toolbox` pre-installed at [drozzoff/xsuite](https://hub.docker.com/r/drozzoff/xsuite).

Pull the latest image with:
```bash
docker pull drozzoff/xsuite:latest-gsihpc
```