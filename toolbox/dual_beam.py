import xtrack as xt
import numpy as np
from numpy.typing import NDArray
from scipy.constants import c as clight
from collections.abc import Callable


def create_two_lines(
	reference_line: xt.Line | Callable,
	rigidity: NDArray[np.floating],
	timestamps: NDArray[np.floating],
	ions: list[xt.Particles],
	harmonics: list[int],
	cavities: list[str],
	voltage_ramps: list[dict],
) -> list[xt.Line]:
	"""
	Creates 2 beamlines. The first one is set for the first ion, second one is set for the second ion.
	Despite their rigidities match, tracking non-reference particle during acceleration is **not 
	currently possbile**. The non reference beam is not properly clocked during acceleration.
	
	Each line has an energy program attached to them following the rigidity ramp.
	They have RF of the other ion attached too. 
	The cavities phase is calcualted via linear interpolation at the timestamps given.

	Since variables are update once per turn in the `xtrack` traking, the timestep should be smaller
	then the revolution time at any point.

	Parameters
	----------
	reference_line
		A line or a function that returns a line to use
	rigidity
		Rigidity ramp
	timestamps
		Timestamps of the rigidity ramp
	ions
		Ions used as reference particles in each line
	harmonics
		Harmonic numbers for each RF
	cavities
		Names of the cavities used for each ion
	voltage_ramps
		Voltage ramps for the corresponding cavities
	"""

	time_delta = np.diff(timestamps)

	lines = []

	# sampled at timestamps
	lines_frf = []
	lines_rf_phase = []
	
	for ion, h, cavity, voltage_ramp in zip(ions, harmonics, cavities, voltage_ramps):
		line = reference_line() if isinstance(reference_line, Callable) else reference_line.copy()
		line.particle_ref = ion
		line.energy_program = xt.EnergyProgram(t_s = timestamps, p0c = rigidity * abs(ion.q0) * clight)
	
		line_frf = h * line.energy_program.get_frev_at_t_s(timestamps)
		
		line_frf_avg = 0.5 * (line_frf[:-1] + line_frf[1:])
		line_rf_phase_per_turn = 2 * np.pi * line_frf_avg * time_delta
	
		line_rf_phase = np.concatenate(([0.0], np.cumsum(line_rf_phase_per_turn)))
	
		line.functions['f_rf'] = xt.FunctionPieceWiseLinear(x = timestamps, y = line_frf)
		line.functions['rf_phase'] = xt.FunctionPieceWiseLinear(x = timestamps, y = line_rf_phase)
		line.functions['V_rf_ref'] = xt.FunctionPieceWiseLinear(x = voltage_ramp['timestamps'], y = voltage_ramp['V'])
		
		line[cavity].absolute_time = False
		line[cavity].phase = line.functions['rf_phase'](line.ref['t_turn_s'])
		line[cavity].frequency = line.functions['f_rf'](line.ref['t_turn_s'])
		line[cavity].voltage = line.functions['V_rf_ref'](line.ref['t_turn_s'])

		lines.append(line)
		lines_frf.append(line_frf)
		lines_rf_phase.append(line_rf_phase)

	# setting none reference RFs
	for i, line in enumerate(lines):
		for j, (cavity, _f_rf, _phase_rf, voltage_ramp) in enumerate(zip(cavities, lines_frf, lines_rf_phase, voltage_ramps)):
			if i == j:
				continue
			line.functions[f"f_rf_non_ref_{j}"] = xt.FunctionPieceWiseLinear(x = timestamps, y = _f_rf)
			line.functions[f"rf_phase_non_ref_{j}"] = xt.FunctionPieceWiseLinear(x = timestamps, y = _phase_rf)
			line.functions[f"V_rf_ref_non_ref_{j}"] = xt.FunctionPieceWiseLinear(x = voltage_ramp['timestamps'], y = voltage_ramp['V'])
			
			line[cavity].absolute_time = False
			line[cavity].phase = line.functions[f"rf_phase_non_ref_{j}"](line.ref['t_turn_s'])
			line[cavity].frequency = line.functions[f"f_rf_non_ref_{j}"](line.ref['t_turn_s'])
			line[cavity].voltage = line.functions[f"V_rf_ref_non_ref_{j}"](line.ref['t_turn_s'])
			
			
	return lines