import numpy as np
from numpy.typing import NDArray


def adjust_Bramp(
	rigidity_init: float,
	rigidity_final: float,
	rho: float,
	Bramp: float,
	ramp_start: float,
	t_rounding: float,
):
	"""
	Adjusts the ramp duration so that its duration has a full number of miliseconds.
	
	Does int() not the roudning
	
	Parameters
	----------
	rigidity_init
		Rigidity at the start of the ramp [Tm]
	rigidity_final
		Rigidity at the end of the ramp [Tm]
	rho
		Bending radius [m]
	Bramp
		Reference B ramp speed [T/s]
	ramp_start
		Moment in time the ramp statrs [s]
	t_rounding
		Rounding time at the start/end of the ramp [s]
	Returns
	-------
	ramp_stop : float
		Moment of the time the B ramp is finished [s]
	Bramp_adjusted : float
		Adjusted B ramp speed. [T/s]
	"""
	# making the ramp_stop time to be compliable with 1 milisecond step

	print(f"Initial ramp stop {ramp_start + t_rounding + (rigidity_final - rigidity_init) / (Bramp * rho)} s")
	ramp_stop_int = ((ramp_start + t_rounding + (rigidity_final - rigidity_init) / (Bramp * rho)) // 1e-3) * 1e-3
	print(f"Cut to {ramp_stop_int} s")
	
	Bramp_adjusted = (rigidity_final - rigidity_init) / (rho * (ramp_stop_int - ramp_start - t_rounding))

	print(f"B-ramp speed was adjusted to {(Bramp_adjusted):.5f} T/s; Ramp ends at {ramp_stop_int} s")
	return ramp_stop_int, Bramp_adjusted

def get_rigidity_ramp(
	t: NDArray[np.floating],
	*,
	rigidity_init: float,
	rigidity_final: float,
	rho: float,
	Bramp: float,
	ramp_start: float,
	t_rounding: float,
) -> NDArray[np.float32]:
	"""
	Evaluates the rigidity ramp according the the provided parameters.
	The format is the same as in SIS18 synchrotron
		
	Parameters
	----------
	t
		Timestamps for the ramp evaluation [s]
	rigidity_init
		Rigidity at the start of the ramp [Tm]
	rigidity_final
		Rigidity at the end of the ramp [Tm]
	rho
		Bending radius [m]
	Bramp
		Reference B ramp speed [T/ms]
	ramp_start
		Moment in time the ramp statrs [s]
	t_rounding
		Rounding time at the start/end of the ramp [s]
	Returns
	-------
	rigidity_ramp
		An array with the rigidities evaluated at given timestamps
	"""
	ramp_stop, Bramp_adjusted = adjust_Bramp(rigidity_init, rigidity_final, rho, Bramp, ramp_start, t_rounding)

	B0 = rigidity_init / rho

	# B at the end of the initial roudning
	B_after_init_rounding = B0 + Bramp_adjusted * t_rounding / 2

	# B at the end of the straight section
	B_after_linear = B_after_init_rounding + Bramp_adjusted * (ramp_stop - ramp_start - 2 * t_rounding)

	# B at the end of the ramp
	B_after_final_rounding = B_after_linear + Bramp_adjusted * t_rounding / 2

	# before the ramp starts
	B_before_ramp = np.where(t < ramp_start, B0, 0.0)
	
	# within the initial rounding
	B_init_rounding = np.where((t >= ramp_start) & (t < ramp_start + t_rounding), B0 + Bramp_adjusted * (t - ramp_start)**2 / (2 * t_rounding), 0.0)

	# at the linear ramp
	B_linear_ramp = np.where((t >= ramp_start + t_rounding) & (t < ramp_stop - t_rounding), B_after_init_rounding + Bramp_adjusted * (t - ramp_start - t_rounding), 0.0)

	# at the final rounding
	B_final_rounding = np.where((t >= ramp_stop - t_rounding) & (t < ramp_stop), B_after_linear + Bramp_adjusted * (t - ramp_stop + t_rounding) - Bramp_adjusted / (2 * t_rounding) * (t - ramp_stop + t_rounding)**2, 0.0)
	
	# after the ramp is done
	B_after_ramp = np.where(t >= ramp_stop, B_after_final_rounding, 0.0)
	
	B = B_before_ramp + B_init_rounding + B_linear_ramp + B_final_rounding + B_after_ramp
	
	return B * rho

