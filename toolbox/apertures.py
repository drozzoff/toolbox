from xtrack import Line, LimitRect, LimitEllipse
from warnings import warn

# default value for the upper and lower jaws when not used
UNLIMITED = 1e10
BEAMPIPE_LIMIT = 0.150

def realign_mad_apertures(line: Line, suppress_output = True):

	for name in line.element_dict:
		element = line.element_dict[name]
		try:
			if element.shift_x != 0.0:
				if isinstance(element, LimitRect):
					if not suppress_output:
						print(f"{element.__class__.__name__}, name = '{name}', shift_x = {element.shift_x}\n\t min_x = {element.min_x}, max_x = {element.max_x}, min_y = {element.min_y}, max_y = {element.max_y}")
					
					# modifying the apertures wrt to the offsets
					element.min_x += element.shift_x # lower jaw
					element.max_x += element.shift_x # upper jaw

					# removing 1 of the jaws when it is too larger (abs > 150 mm)
					if abs(element.min_x) > BEAMPIPE_LIMIT:
						element.min_x = -UNLIMITED
					if abs(element.max_x) > BEAMPIPE_LIMIT:
						element.max_x = UNLIMITED

					element.shift_x = 0.0
					
				elif isinstance(element, LimitEllipse):
					warn("The elliptic aperture is missaligned!")
					if not suppress_output:
						print(f"{element.__class__.__name__}, name = '{name}', shift_x = '{element.shift_x}'\n\t a = {element.a}, b = {element.b}")
				else:
					raise Exception("Element is not aperture element")
		except AttributeError:
			pass