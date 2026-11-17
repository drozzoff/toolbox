# Xsuite Toolbox

Research utilities built around [Xsuite](https://xsuite.readthedocs.io/) for
SIS18 beam dynamics, slow extraction, particle tracking, diagnostics, and
visualization.

[Separatrix analysis](guides/separatrix_search.md){ .md-button .md-button--primary }
[Set up multi-GPU tracking](guides/multi_gpu.md){ .md-button }

## Start from your workflow

<div class="grid cards" markdown>

-   **Analyze a separatrix**

    ---

    Locate the stable-region boundary and calculate the enclosed area, spiral
    step, and extraction angle.

    [Open the separatrix guide →](guides/separatrix_search.md)

-   **Match extraction properties**

    ---

    Use separatrix properties as Xtrack matching targets or evaluate them in a
    parallel optimization workflow.

    [See the matching workflow →](guides/separatrix_matching.md)

-   **Track on multiple GPUs**

    ---

    Distribute particles across OpenCL or Cuda devices, record phase-space snapshots,
    and save checkpoints during long tracking jobs.

    [Open the multi-GPU guide →](guides/multi_gpu.md)

</div>

## Guides and API references

The API reference is generated from the package source, so signatures,
parameters, and defaults stay close to the implementation.

<div class="grid cards" markdown>

-   **Separatrix API**

    `get_separatrix_properties()`, `get_separatrix_vertices()`, `get_stable_and_unstable_particle()`, 
	and `ActionSeparatrix`.

    [Browse the separatrix API →](reference/separatrix.md)

-   **Multi-GPU API**

    Reference documentation for `track_multigpu()` and its tracking options.

    [Browse the multi-GPU API →](reference/multi_gpu.md)

</div>

## More in the package

The toolbox also contains utilities for phase-space snapshots, multi-species
beamlines, SIS18 tune and excitation analysis, rigidity ramps, lattice-aware
plotting, and slow-extraction dashboard profiles. These parts of the package
are still being documented; the focused guides above are the best starting
point for the currently documented workflows.

!!! note "Research software"

    The toolbox is under active development, and its API may change between
    versions.

## Installation

Install a local checkout with Python 3.10 or newer:

```bash
git clone https://github.com/drozzoff/toolbox.git
cd toolbox
python -m pip install .
```

For development, use `python -m pip install -e .` instead.
