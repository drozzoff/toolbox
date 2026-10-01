from warnings import warn
import numpy as np
import xtrack as xt

# default value for the upper and lower jaws when not used
UNLIMITED = 1e10
BEAMPIPE_LIMIT = 0.150

def realign_mad_apertures(line: xt.Line, suppress_output = True):
	"""
	Realigns to non-symmetric apertures that were defined in MAD-X
	via a loophole (big symmetric aperture misaligned transversely).

	It has a hardcoded limit `BEAMPIPE_LIMIT = 0.15` for the maximum
	absolute aperture in either direction. The larger apertures are
	set to a very large value (as used in `xsuite`).
	"""
	for name in line.element_dict:
		element = line.element_dict[name]

		if isinstance(element, xt.LimitRect):

			if element.shift_x != 0.0:
				if not suppress_output:
					print(f"{element.__class__.__name__}, name = '{name}', \
						shift_x = {element.shift_x}\n\t min_x = {element.min_x}, \
						max_x = {element.max_x}"
					)

				# modifying the apertures wrt to the offsets
				element.min_x += element.shift_x # lower jaw
				element.max_x += element.shift_x # upper jaw

				# removing 1 of the jaws when it is too large (abs > 150 mm)
				if abs(element.min_x) > BEAMPIPE_LIMIT:
					element.min_x = -UNLIMITED
				if abs(element.max_x) > BEAMPIPE_LIMIT:
					element.max_x = UNLIMITED

				element.shift_x = 0.0

			if element.shift_y != 0.0:
				if not suppress_output:
					print(f"{element.__class__.__name__}, name = '{name}', \
						shift_y = {element.shift_y}\n\t min_y = {element.min_y}, \
						max_y = {element.max_y}"
					)

				# modifying the apertures wrt to the offsets
				element.min_y += element.shift_y # lower jaw
				element.max_y += element.shift_y # upper jaw

				# removing 1 of the jaws when it is too large (abs > 150 mm)
				if abs(element.min_y) > BEAMPIPE_LIMIT:
					element.min_y = -UNLIMITED
				if abs(element.max_y) > BEAMPIPE_LIMIT:
					element.max_y = UNLIMITED

				element.shift_y = 0.0

		elif isinstance(element, xt.LimitEllipse):
			if element.shift_x != 0.0 or element.shift_y != 0.0:
				warn("The elliptic aperture is misaligned!")
				if not suppress_output:
					print(f"{element.__class__.__name__}, name = '{name}', \
						shift_x = '{element.shift_x}', shift_y = '{element.shift_y}', \
						\n\t a = {element.a}, b = {element.b}")


def _remove_inactive_multipoles_fix(line: xt.Line):
	"""
	Function to replace inactive thick mutipoles.
	Is needed because `Line.optimize_for_tracking()` does not handle them well.
	"""

	for ele, ele_name in zip(line, line.element_names):
		if isinstance(ele, xt.Multipole):
			aux = ([ele.hxl] + list(ele.knl) + list(ele.ksl))
			if np.sum(np.abs(np.array(aux))) == 0.0:
				if ele.isthick and ele.length != 0:
					line.remove(ele_name)

				
			
