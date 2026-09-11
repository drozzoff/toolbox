from warnings import warn
import numpy as np
import xtrack as xt

# default value for the upper and lower jaws when not used
UNLIMITED = 1e10
BEAMPIPE_LIMIT = 0.150

def realign_mad_apertures(line: xt.Line, suppress_output = True):
	"""
	Realignes to non symmetric apertures that are defined in MAD-X
	as symmetric aperture with offset.
	"""
	for name in line.element_dict:
		element = line.element_dict[name]
		try:
			if element.shift_x != 0.0:
				if isinstance(element, xt.LimitRect):
					if not suppress_output:
						print(f"{element.__class__.__name__}, name = '{name}', \
							shift_x = {element.shift_x}\n\t min_x = {element.min_x}, \
							max_x = {element.max_x}, min_y = {element.min_y}, max_y = {element.max_y}"
						)
					
					# modifying the apertures wrt to the offsets
					element.min_x += element.shift_x # lower jaw
					element.max_x += element.shift_x # upper jaw

					# removing 1 of the jaws when it is too larger (abs > 150 mm)
					if abs(element.min_x) > BEAMPIPE_LIMIT:
						element.min_x = -UNLIMITED
					if abs(element.max_x) > BEAMPIPE_LIMIT:
						element.max_x = UNLIMITED

					element.shift_x = 0.0
					
				elif isinstance(element, xt.LimitEllipse):
					warn("The elliptic aperture is missaligned!")
					if not suppress_output:
						print(f"{element.__class__.__name__}, name = '{name}', \
							shift_x = '{element.shift_x}'\n\t a = {element.a}, b = {element.b}")
				else:
					raise Exception("Element is not aperture element")
		except AttributeError:
			pass

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

				
			
