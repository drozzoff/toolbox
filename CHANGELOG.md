# Changelog

## [0.1.4b] - 06-11-2026

### Changed

- Better error hanlding in the `get_separatrix_properties()` during optimizations.


## [0.1.4a] - 05-11-2026

### Changed

- Modified the default number of particles
- Updated documentation

## [0.1.4] - 04-11-2026

### Changed

- Modified functions related to separatrix search/properties to accept cached twiss.
- Separatrix search is done using the lost particles.
- `get_separatrix_properties()` is more robust to fails and can be used with optimizers.

### Added

- `ActionSeparatrix` that can be used as action in `xtrack` matching

## [0.1.3] - 03-11-2026

### Added

- A new function `get_separatrix_properties()` that returns the separatrix properties.

### Changed

- Adjusted the API for `get_stable_and_unstable_particle()`
- Separated internally the separatrix search and separatrix geometry

## [0.1.2] - 19-10-2026

### Changed

- Introduced automatic Documentation deployment with MKDocs and GitHub Actions.
- `toolbox.track_multigpu()` now supports tracking with checkpoints.
- `toolbox.multigpu.save_monitor_portion()` now saves `particle_id` and `at_element`

## [0.1.1] - 01-10-2026

### Changed

- Bug fixes in `toolbox.dashboard_profiles`
- `toolbox.visualisation.Plotcontext` now can plot vertical apertures too.
- `toolbox.realign_mad_apertures()` now realigns vertical rectangular apertures too.

## [0.1.0] - 16-09-2026

First version release.