import xtrack as xt
import numpy as np
import ipywidgets as widgets
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display


def phase_space_evolution(line: xt.Line, monitor_name: str, particles: xt.Particles, **kwargs):
	"""
	"""
	def get_phase_portraint(at_turn = 0):

		# if particle is lost before `at_turn` returns the turn it was lost at.
		# Otherwise returns `at_turn`
		
		at_turn_flat = list(map(lambda x: at_turn if x > at_turn else x, particles.at_turn))
			
		x_res = np.array(list(map(lambda x, y: x[y], line.element_dict[monitor_name].x, at_turn_flat)))
		px_res = np.array(list(map(lambda x, y: x[y], line.element_dict[monitor_name].px, at_turn_flat)))
		
		return x_res, px_res

	# masks
	septum_loss = abs(particles.s - line.get_s_position(monitor_name)) < 1e-6

	_lost = particles.state == 0.0

	lost = lambda at_turn: particles.at_turn < at_turn

	lost_on_septum = lambda turn: np.logical_and(septum_loss, lost(turn))

	alive_in_last_100_turns = lambda at_turn: particles.at_turn + 100 > at_turn

	lost_in_last_100_turns = lambda turn: np.logical_and(lost(turn), alive_in_last_100_turns(turn))

	lost_on_septum_in_last_100_turns = lambda turn: np.logical_and(septum_loss, lost_in_last_100_turns(turn))

	toggle_btn = widgets.ToggleButton(
		value = False,
		description = 'Pause Animation',
		disabled = False,
		button_style = '', 
		tooltip = 'Pause/Resume Animation',
	)

	fig, axes = plt.subplots()

	def update(at_turn):
		axes.cla()
		
		x_at_turn, px_at_turn = get_phase_portraint(at_turn)
		sns.scatterplot(
			x = x_at_turn[~lost(at_turn)], 
			y = px_at_turn[~lost(at_turn)],
			color = "green", 
			alpha = 0.6, 
			s = 5,
			ax = axes,
			label = "Alive"
		)

		sns.scatterplot(
			x = x_at_turn[lost_on_septum_in_last_100_turns(at_turn)], 
			y = px_at_turn[lost_on_septum_in_last_100_turns(at_turn)],
			color = "red", 
			alpha = 0.6, 
			s = 5,
			ax = axes,
			label = "Lost at monitor location"
		)
		
		axes.set_xlabel("x [m]")
		axes.set_ylabel("x'")
		
		axes.set_title(f"Monitor '{monitor_name}', turn = {at_turn}")
		
		axes.axvline(x = -0.055, color = "grey", dashes = (5, 2), label = "ES septum aperture")
		
		axes.legend()

		plt.ylim(-0.011, 0.005)
		plt.xlim(-0.09, 0.06)

	# Create animation
	ani = FuncAnimation(
		fig, 
		update, 
		frames = np.arange(0, 2000, 3),  # Adjust step size for speed
		interval = 50  # Time between frames in ms
	)
	plt.show()

	# A flag to track animation state
	is_running = True

	def on_toggle_change(change):
		global is_running
		if change['new']:
			ani.event_source.stop()
			toggle_btn.description = 'Resume Animation'
			is_running = False
		else:
			ani.event_source.start()
			toggle_btn.description = 'Pause Animation'
			is_running = True

	toggle_btn.observe(on_toggle_change, names = 'value')
	display(toggle_btn)
