import matplotlib
import sys
import matplotlib.pyplot as plt
import json
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Polygon
import numpy as np
from rich.console import Console
from IPython.display import display

import xtrack as xt

def is_notebook() -> bool:
	try:
		shell = get_ipython().__class__.__name__
		return shell == 'ZMQInteractiveShell'
	except NameError:
		return False

def get_thick_element_length(line: xt.Line, base_name: str) -> float:
	# there are 2 types of elements we are looking
	# with a name f"drift_{base_name}.." -> right and left drift in 
	#    drift-kick-drift form
	# with a name base_name -> this is our actual element
	#
	# Sometimes the base element has a length property, while right 
	#    and left drifts are created. We have to take either sum of drifts
	#    or the length of the element.
	base_element = line.element_dict[base_name]

	if hasattr(base_element, "length"):
		# element has a lemgth property, ignoring drifts
		return base_element.length
	else:
		# element has no length property, summin up 2 drifts
		return sum(map(lambda x: line[x].length, filter(lambda x: f"drift_{base_name}.." in x, line.element_names)))

def get_orientation(element):
	if isinstance(element, xt.Quadrupole):
		if element.k1 != 0.0:
			return element.k1 / abs(element.k1)
		else:
			return None
	if isinstance(element, xt.Sextupole):
		if element.k2 != 0.0:
			return element.k2 / abs(element.k2)
		else:
			return None
	return None

class PlotContext:
	"""
	The class that holds the context of the plot with a precompiled background
	including the survey and apertures.
	"""
	def __init__(self, 
			  style: str = "style.json",
			  show_survey: bool = True,
			  show_apertures: bool = True,
			  line: xt.Line = None
			  ):
		"""
		Paremeters
		----------
		style : str
			Path to the style file.
		show_survey : bool
			Display the survey on top of the plot or not.
		show_apertures : bool
			Display the apertures on the top and bottom of the plot or not.
		line : xt.Line
			The line object to plot the survey and apertures.
		"""
		self.in_notebook = is_notebook()
		plt.ioff()

		with open(style) as f:
			self.config = json.load(f)
		
		self.fig = plt.figure(figsize = (self.config['Figure_size']['width'], self.config['Figure_size']['height']))

		self.console = Console()
		self._show_survey, self._show_apertures = False, False

		if (show_apertures or show_survey) and line is None:
			raise ValueError("xt.Line object is needed to plot survey and/or apertures.")

		self.line = line
		self.dynamic_figures = []
		self.set_figure()

		self.show_survey = show_survey
		self.show_apertures = show_apertures

	def set_figure(self):
		figure_height = self.config['Figure_size']['height']
		
		self.fig.clf()

		if self._show_survey:
			self.gs = GridSpec(2, 1, height_ratios = [self.config['Survey']['Height'], figure_height - self.config['Survey']['Height']])
			self.survey_subplot = self.fig.add_subplot(self.gs[0])
			self.main_subplot = self.fig.add_subplot(self.gs[1], sharex = self.survey_subplot)
		else:
			self.gs = GridSpec(1, 1)
			self.main_subplot = self.fig.add_subplot(self.gs[0])

	def need_apertures_warmup(self):
		return self._show_apertures and (not hasattr(self, '_apertures_warmed') or not self._apertures_warmed)

	def need_survey_warmup(self):
		return self._show_survey and (not hasattr(self, '_survey_warmed') or not self._survey_warmed)

	@property
	def show_survey(self):
		return self._show_survey

	@show_survey.setter
	def show_survey(self, value: bool):
		if value != self._show_survey:
			self._show_survey = value
			self.set_figure()

			# hard reset, because adding survey changes the layout
			self._survey_warmed = False
			self._apertures_warmed = False

			if self.need_survey_warmup():
				self.warm_survey()
				self._survey_warmed = True

			if self.need_apertures_warmup():
				self.warm_apertures()
				self._apertures_warmed = True

	@property
	def show_apertures(self):
		return self._show_apertures

	@show_apertures.setter
	def show_apertures(self, value: bool):
		if value != self._show_apertures:
			self._show_apertures = value

			if self.need_apertures_warmup():
				self.warm_apertures()
				self._apertures_warmed = True

			for artist in self.aperture_artists:
				artist.set_visible(self._show_apertures)

#	@console_log
	def warm_survey(self):
		"""
		Plot the survey and save it to use as a background later.
		"""
		self.survey_artists = []

		curve, = self.survey_subplot.plot(np.linspace(0.0, self.line.get_length(), 100), [0.0] * 100, '-', color = "black", linewidth = 1.0)

		self.survey_artists.append(curve)
		
		self.survey_subplot.set_ylim(self.config['Survey']['y_min'], self.config['Survey']['y_max'])
		for s, element in zip(self.line.get_table().s, self.line.elements):
			elem_type = element.__class__.__name__
		
			if elem_type in self.config['Survey']['ElementsToPlot'] and element.length != 0.0:
				try:
					style = self.config['Survey']['ElementStyles'][elem_type]
				except KeyError:
					style = self.config['Survey']['ElementStyles']['Default']

				orientation = get_orientation(element)
				if style['orientation'] and orientation is not None:
					rectangle = Rectangle((s, 0), element.length, style['height'] * orientation, linewidth = 0.0, color = style['color'])
					self.survey_subplot.add_patch(rectangle)
					self.survey_artists.append(rectangle)
				else:
					rectangle = Rectangle((s, -style['height'] / 2), element.length, style['height'], linewidth = 0.0, color = style['color'], alpha = 0.5)
					self.survey_subplot.add_patch(rectangle)
					self.survey_artists.append(rectangle)

		self.survey_subplot.set_yticks([])

#	@console_log
	def warm_apertures(self, **kwargs):
		"""
		Plot the aperture and save it to use as a background later.
		"""
		elliptic_aper_cache = {'s_up': [], 'aper_up': [], 's_down': [], 'aper_down': [], 'is_empty': True}
		
		self.aperture_artists = []

		for i, name in enumerate(self.line.element_dict):
			element = self.line.element_dict[name]
			s = self.line.get_s_position(name)

			if isinstance(element, xt.LimitEllipse):
				# that means the name of the element the this aperture
				# belongs is
				base_name = name.replace('_aper', '')
		
				# exctracting the element and any associated drifts
				length = get_thick_element_length(self.line, base_name)

				# adding the first point a dict is empty
				if elliptic_aper_cache['is_empty']:
					elliptic_aper_cache['s_up'].append(s)
					elliptic_aper_cache['s_down'].append(s)
					
					elliptic_aper_cache['aper_up'].append(self.config['Aperture']['Beampipe']['x'])
					elliptic_aper_cache['aper_down'].append(-self.config['Aperture']['Beampipe']['x'])
				
				# while the aperture is elliptic, filling up the dict
				elliptic_aper_cache['s_up'].extend([s, s + length])
				elliptic_aper_cache['s_down'].extend([s, s + length])

				elliptic_aper_cache['aper_up'].extend([element.a, element.a])
				elliptic_aper_cache['aper_down'].extend([-element.a, -element.a])

				elliptic_aper_cache['is_empty'] = False   
							
			if isinstance(element, xt.LimitRect):

				# if the aperture is Rectangular and there is some elliptic data
				# not plotted then plotting it and ressseting elliptic_aper_cache
				# + adding a drop to a beamline aperture
				
				if not elliptic_aper_cache['is_empty']:
					# drop down to the default value
					elliptic_aper_cache['s_up'].append(elliptic_aper_cache['s_up'][-1])
					elliptic_aper_cache['s_down'].append(elliptic_aper_cache['s_down'][-1])
					
					elliptic_aper_cache['aper_up'].append(self.config['Aperture']['Beampipe']['x'])
					elliptic_aper_cache['aper_down'].append(-self.config['Aperture']['Beampipe']['x'])
					
					curve_up, = self.main_subplot.plot(elliptic_aper_cache['s_up'], elliptic_aper_cache['aper_up'], '-', color = "red")
					curve_down, = self.main_subplot.plot(elliptic_aper_cache['s_down'], elliptic_aper_cache['aper_down'], '-', color = "red")
					
					self.aperture_artists.extend([curve_up, curve_down])

					#reseting the dict
					for key in elliptic_aper_cache:
						elliptic_aper_cache[key] = []
						elliptic_aper_cache['is_empty'] = True
				
				base_name = name.replace('_aper', '')
				length = get_thick_element_length(self.line, base_name)

				if base_name == "gs04me1e":
					# Custom plotting for the septum
					es_angle = kwargs.get('es_angle', -7.4e-3)
					polygon = Polygon([[s, -0.055], [s + length, -0.055 + length * es_angle],
									   [s + length, -self.config['Aperture']['Beampipe']['x']], [s, -self.config['Aperture']['Beampipe']['x']]], color = "black", linewidth = 1.0)
					self.main_subplot.add_patch(polygon)
					self.aperture_artists.append(polygon)

					
					if element.max_x > self.config['Aperture']['Rectangular']['limit_x']:
						curve, = self.main_subplot.plot([s, s + length], [self.config['Aperture']['Beampipe']['x'], self.config['Aperture']['Beampipe']['x']], '-', color = "black")
						self.aperture_artists.append(curve)
					else:
						rectangle = Rectangle((s, self.config['Aperture']['Beampipe']['x']), length, element.max_x - self.config['Aperture']['Beampipe']['x'], color = "black", linewidth = 1.0)
						self.main_subplot.add_patch(rectangle)
						self.aperture_artists.append(rectangle)
					
					continue


				if element.min_x < -self.config['Aperture']['Rectangular']['limit_x']:
					curve, = self.main_subplot.plot([s, s + length], [-self.config['Aperture']['Beampipe']['x'], -self.config['Aperture']['Beampipe']['x']], '-', color = "black")
					self.aperture_artists.append(curve)
				else:
					rectangle = Rectangle((s, -self.config['Aperture']['Beampipe']['x']), length, element.min_x + self.config['Aperture']['Beampipe']['x'], color = "black", linewidth = 1.0)
					self.main_subplot.add_patch(rectangle)
					self.aperture_artists.append(rectangle)

				if element.max_x > self.config['Aperture']['Rectangular']['limit_x']:
					curve, = self.main_subplot.plot([s, s + length], [self.config['Aperture']['Beampipe']['x'], self.config['Aperture']['Beampipe']['x']], '-', color = "black")
					self.aperture_artists.append(curve)
				else:
					rectangle = Rectangle((s, self.config['Aperture']['Beampipe']['x']), length, element.max_x - self.config['Aperture']['Beampipe']['x'], color = "black", linewidth = 1.0)
					self.main_subplot.add_patch(rectangle)
					self.aperture_artists.append(rectangle)
		
	def add_plot(self, x, y, *args, **kwargs):
		"""
		Add a plot to the main subplot.
		"""
		curve,  = self.main_subplot.plot(x, y, *args, **kwargs)
		self.dynamic_figures.append(curve)


	def __enter__(self):
		"""Enter context: Restore the background and return the axis for plotting."""
		for curve in self.dynamic_figures:
			try:
				curve.remove()
			except Exception:
				pass

		self.dynamic_figures = []
		self.fig.canvas.draw_idle()
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		"""Exit context: Update the plot efficiently."""
		
		if self.in_notebook:
			self.fig.canvas.draw_idle()
			display(self.fig)

	def commit(self):
		"""Show the plot in terminal mode"""
		for artist in self.dynamic_figures:
			try:
				artist.remove()
			except Exception:
				pass
		self.dynamic_figures = []
		# Update the canvas to reflect the removal.
		self.fig.canvas.draw_idle()